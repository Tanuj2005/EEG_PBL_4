"""
Lightweight EEG Emotion Recognition Model:
  1. Depthwise Temporal Conv
  2. Micro Graph Layer (fixed adjacency)
  3. Tiny Attention (SE-style)
  4. Emotion Classifier

Total parameters < 20k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


def build_adjacency_matrix() -> torch.Tensor:
    """
    Build fixed adjacency matrix for micro-graph.
    Nodes = 7 (hemisphere pairs).

    Connectivity:
      - Self-loops
      - Adjacent pairs (frontal-central-temporal-parietal-occipital chain)
      - Cross-region links (frontal↔occipital, temporal↔parietal)

    Returns:
        adj: [7 x 7] normalized adjacency matrix
    """
    num_nodes = config.NUM_PAIRS  # 7
    # Pair indices:
    # 0: AF3-AF4 (frontal)
    # 1: F7-F8   (frontal)
    # 2: F3-F4   (frontal)
    # 3: FC5-FC6 (fronto-central)
    # 4: T7-T8   (temporal)
    # 5: P7-P8   (parietal)
    # 6: O1-O2   (occipital)

    adj = np.eye(num_nodes, dtype=np.float32)  # Self-loops

    # Adjacent connections (chain along scalp)
    adjacent_edges = [
        (0, 1), (0, 2), (1, 2), (1, 4),  # frontal cluster
        (2, 3), (3, 4),                    # central
        (4, 5), (5, 6),                    # posterior chain
    ]

    # Cross-region links (neuroscience-inspired)
    cross_edges = [
        (0, 6),  # frontal ↔ occipital
        (2, 5),  # F3/F4 ↔ P7/P8
        (3, 5),  # FC5/FC6 ↔ P7/P8
    ]

    for (i, j) in adjacent_edges + cross_edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Normalize: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    adj_normalized = d_inv_sqrt @ adj @ d_inv_sqrt

    return torch.tensor(adj_normalized, dtype=torch.float32)


class DepthwiseTemporalConv(nn.Module):
    """
    Depthwise Separable 1D Convolution for temporal modeling.
    Operates on the feature dimension per node.

    Input:  [batch, channels, length]
    Output: [batch, channels, length]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        # Depthwise conv: each input channel convolved independently
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        # Pointwise conv: 1x1 to mix channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroGraphLayer(nn.Module):
    """
    Lightweight Graph Convolution with fixed adjacency.
    GCN: H' = σ(A_norm @ H @ W)

    Input:  [batch, num_nodes, features]
    Output: [batch, num_nodes, out_features]
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ELU(inplace=True)

        # Fixed adjacency (not learnable)
        self.register_buffer('adj', build_adjacency_matrix())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_nodes, features]
        Returns:
            [batch, num_nodes, out_features]
        """
        # Graph convolution: aggregate neighbors
        # adj: [num_nodes, num_nodes], x: [B, N, F]
        x = torch.matmul(self.adj, x)  # [B, N, F]
        x = self.weight(x)              # [B, N, out_features]

        # BatchNorm across feature dim
        B, N, F = x.shape
        x = x.reshape(B * N, F)
        x = self.bn(x)
        x = x.reshape(B, N, F)

        x = self.act(x)
        return x


class TinyAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Ultra-lightweight: only 2 * (C * C/r) parameters.

    Input:  [batch, channels]
    Output: [batch, channels] (re-weighted)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels]
        Returns:
            [batch, channels]
        """
        attn = self.fc(x)  # [B, C]
        return x * attn


class ResidualBlock(nn.Module):
    """
    Residual block with GELU activation and BatchNorm.
    Input and output have the same dimension so the skip connection works.
    """

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class EmotionClassifier(nn.Module):
    """
    Complete lightweight model with residual connections.
    """

    def __init__(self,
                 num_pairs: int = config.NUM_PAIRS,
                 features_per_pair: int = config.FEATURES_PER_PAIR,
                 temporal_channels: int = config.TEMPORAL_CHANNELS,
                 graph_hidden: int = config.GRAPH_HIDDEN,
                 attention_reduction: int = config.ATTENTION_REDUCTION,
                 num_classes: int = config.NUM_CLASSES_VALENCE,
                 dropout: float = 0.3):
        super().__init__()

        self.num_pairs = num_pairs
        self.features_per_pair = features_per_pair

        # --- Stage 1: Feature projection ---
        self.feature_proj = nn.Sequential(
            nn.Linear(features_per_pair, temporal_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Stage 1b: Temporal conv ---
        self.temp_conv = DepthwiseTemporalConv(
            num_pairs, num_pairs, kernel_size=min(config.TEMPORAL_KERNEL, temporal_channels)
        )

        # --- Stage 2: Micro Graph Layers ---
        self.graph1 = MicroGraphLayer(temporal_channels, graph_hidden)
        self.graph2 = MicroGraphLayer(graph_hidden, graph_hidden)
        self.graph_dropout = nn.Dropout(dropout)

        # --- Stage 2b: Residual block after graph (operates on pooled features) ---
        self.residual = ResidualBlock(graph_hidden, dropout=dropout)

        # --- Stage 3: Tiny Attention ---
        pooled_dim = graph_hidden
        self.attention = TinyAttention(pooled_dim, reduction=attention_reduction)

        # --- Stage 4: Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, num_classes),
        )

        # Remove old unused layer
        # self.temporal_conv and self.temporal_to_graph are no longer needed

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear/conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Stage 1: Feature projection per node
        h = self.feature_proj(x)        # [B, 7, temporal_channels]

        # Stage 1b: Temporal conv
        h = self.temp_conv(h)           # [B, 7, temporal_channels]

        # Stage 2: Graph convolution
        h = self.graph1(h)              # [B, 7, graph_hidden]
        h = self.graph_dropout(h)
        h = self.graph2(h)              # [B, 7, graph_hidden]

        # Stage 3: Global pooling
        h = h.mean(dim=1)              # [B, graph_hidden]

        # Stage 3b: Residual refinement
        h = self.residual(h)           # [B, graph_hidden]

        # Stage 3c: Attention
        h = self.attention(h)          # [B, graph_hidden]

        # Stage 4: Classify
        logits = self.classifier(h)    # [B, num_classes]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = config.NUM_CLASSES_VALENCE) -> EmotionClassifier:
    model = EmotionClassifier(num_classes=num_classes)
    print(f"Model created with {model.count_parameters():,} trainable parameters.")
    return model


if __name__ == "__main__":
    model = create_model(num_classes=2)
    print(model)
    dummy = torch.randn(8, 7, 6)
    out = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
    print(f"Total parameters: {model.count_parameters():,}")