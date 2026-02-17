"""
Temporal Graph State Machine (TGSM) for EEG emotion recognition.

Key novelties:
1. Edge State Matrix (ESM) — a recurrent state on the graph structure itself (edges),
   not just node features. Gives the model "graph memory".
2. GRU-style gating on the adjacency/edge matrix evolution.
3. Eigenvalue-based graph spectral readout borrowed from chemistry GNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeStateGRU(nn.Module):
    """
    GRU-style gate for updating the Edge State Matrix (ESM).
    
    ESM_t = (1 - z) ⊙ ESM_{t-1} + z ⊙ Ã_t
    where z = σ(W_z · [ESM_{t-1}, Ã_t])
    
    W_z is implemented as a 1×1 convolution over the edge dimension,
    making it extremely parameter-efficient.
    """
    
    def __init__(self):
        super().__init__()
        # 1x1 conv: input has 2 channels (ESM_{t-1}, Ã_t), output 1 channel (gate z)
        self.gate_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            bias=True
        )
        # Initialize bias to negative value so gate starts near 0 (conservative updates)
        nn.init.constant_(self.gate_conv.bias, -1.0)
        nn.init.xavier_uniform_(self.gate_conv.weight)
    
    def forward(self, esm_prev, adj_candidate):
        """
        Args:
            esm_prev: (batch, N, N) previous edge state matrix
            adj_candidate: (batch, N, N) new candidate adjacency
        
        Returns:
            esm_new: (batch, N, N) updated edge state matrix
        """
        # Stack as 2-channel "image": (batch, 2, N, N)
        stacked = torch.stack([esm_prev, adj_candidate], dim=1)
        
        # Compute gate: (batch, 1, N, N)
        z = torch.sigmoid(self.gate_conv(stacked))
        z = z.squeeze(1)  # (batch, N, N)
        
        # GRU-style update
        esm_new = (1 - z) * esm_prev + z * adj_candidate
        
        return esm_new


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network layer.
    H' = σ(D^{-1/2} A D^{-1/2} H W)
    Uses the ESM as the adjacency matrix.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, node_features, adj):
        """
        Args:
            node_features: (batch, N, in_features)
            adj: (batch, N, N) adjacency matrix (ESM)
        
        Returns:
            out: (batch, N, out_features)
        """
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # Add self-loops
        N = adj.size(-1)
        identity = torch.eye(N, device=adj.device).unsqueeze(0)
        adj = adj + identity
        
        # Degree matrix
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (batch, N, 1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Normalize: D^{-1/2} A D^{-1/2}
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-1, -2)
        
        # Message passing
        support = self.weight(node_features)  # (batch, N, out_features)
        out = torch.bmm(adj_norm, support)  # (batch, N, out_features)
        out = self.norm(out)
        out = F.elu(out)
        
        return out


class SpectralReadout(nn.Module):
    """
    Extract top-k eigenvalues of the final ESM as graph-level spectral features.
    Borrowed from chemistry GNN literature — novel in EEG context.
    """
    
    def __init__(self, k=5):
        super().__init__()
        self.k = k
    
    def forward(self, esm):
        """
        Args:
            esm: (batch, N, N) final edge state matrix
        
        Returns:
            eigenvalues: (batch, k) top-k eigenvalues (sorted descending by magnitude)
        """
        # Make symmetric for real eigenvalues
        esm_sym = (esm + esm.transpose(-1, -2)) / 2
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(esm_sym)  # (batch, N), sorted ascending
        
        # Take top-k by magnitude (last k values since sorted ascending)
        topk = eigenvalues[:, -self.k:]  # (batch, k)
        
        return topk


class TGSM(nn.Module):
    """
    Temporal Graph State Machine for EEG emotion recognition.
    
    Architecture:
    1. For each time window, compute candidate adjacency from DE features (cosine similarity).
    2. Update Edge State Matrix (ESM) with GRU-style gate.
    3. Apply GCN layer using ESM as adjacency.
    4. At trial end, extract spectral features (eigenvalues) + node embeddings.
    5. FC classifier → emotion label.
    
    The key innovation is the recurrent state on the GRAPH STRUCTURE (edges),
    not just node features. This captures "emotional inertia" — the temporal
    momentum of brain connectivity patterns.
    """
    
    def __init__(
        self,
        num_channels=14,
        num_bands=4,
        gcn_hidden=32,
        num_eigenvalues=5,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.num_eigenvalues = num_eigenvalues
        
        # Edge State GRU — the core novelty
        self.edge_gru = EdgeStateGRU()
        
        # GCN layer applied at each time step
        self.gcn = GCNLayer(num_bands, gcn_hidden)
        
        # Spectral readout
        self.spectral_readout = SpectralReadout(k=num_eigenvalues)
        
        # Node-level temporal aggregation (simple attention over time steps)
        self.temporal_attention = nn.Sequential(
            nn.Linear(gcn_hidden, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        # Input: spectral features (k eigenvalues) + pooled node embeddings (gcn_hidden)
        classifier_input_dim = num_eigenvalues + gcn_hidden
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        self._count_parameters()
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TGSM total trainable parameters: {total}")
        
        edge_params = sum(p.numel() for p in self.edge_gru.parameters() if p.requires_grad)
        gcn_params = sum(p.numel() for p in self.gcn.parameters() if p.requires_grad)
        clf_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        print(f"  Edge GRU:    {edge_params}")
        print(f"  GCN layer:   {gcn_params}")
        print(f"  Classifier:  {clf_params}")
    
    def compute_candidate_adjacency(self, de_features):
        """
        Compute candidate adjacency via cosine similarity between node features.
        No learned parameters — purely data-driven.
        
        Args:
            de_features: (batch, N, num_bands) DE features for current window
        
        Returns:
            adj: (batch, N, N) cosine similarity adjacency, values in [0, 1]
        """
        # L2 normalize along feature dimension
        normed = F.normalize(de_features, p=2, dim=-1)  # (batch, N, num_bands)
        
        # Cosine similarity
        adj = torch.bmm(normed, normed.transpose(-1, -2))  # (batch, N, N)
        
        # Map from [-1, 1] to [0, 1]
        adj = (adj + 1) / 2
        
        # Zero out diagonal (no self-loops in candidate adj; added in GCN)
        mask = 1 - torch.eye(self.num_channels, device=adj.device).unsqueeze(0)
        adj = adj * mask
        
        return adj
    
    def forward(self, windows_sequence):
        """
        Process a full trial (sequence of DE feature windows).
        
        Args:
            windows_sequence: (batch, num_windows, num_channels, num_bands)
        
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, num_windows, N, F_dim = windows_sequence.shape
        device = windows_sequence.device
        
        # Initialize ESM to zeros
        esm = torch.zeros(batch_size, N, N, device=device)
        
        # Collect node embeddings from each time step
        all_node_embeddings = []
        
        for t in range(num_windows):
            de_t = windows_sequence[:, t, :, :]  # (batch, N, num_bands)
            
            # Step 1: Compute candidate adjacency (no learned params)
            adj_candidate = self.compute_candidate_adjacency(de_t)  # (batch, N, N)
            
            # Step 2: Update ESM with GRU gate
            esm = self.edge_gru(esm, adj_candidate)  # (batch, N, N)
            
            # Step 3: GCN message passing using current ESM
            node_emb = self.gcn(de_t, esm)  # (batch, N, gcn_hidden)
            
            all_node_embeddings.append(node_emb)
        
        # --- Trial-level readout ---
        
        # Spectral readout from final ESM
        spectral_features = self.spectral_readout(esm)  # (batch, k)
        
        # Temporal attention over node embeddings
        all_node_embeddings = torch.stack(all_node_embeddings, dim=1)  # (batch, T, N, gcn_hidden)
        
        # Pool over nodes first (mean)
        graph_embeddings = all_node_embeddings.mean(dim=2)  # (batch, T, gcn_hidden)
        
        # Attention-weighted temporal aggregation
        attn_weights = self.temporal_attention(graph_embeddings)  # (batch, T, 1)
        pooled = (graph_embeddings * attn_weights).sum(dim=1)  # (batch, gcn_hidden)
        
        # Concatenate spectral + node features
        combined = torch.cat([spectral_features, pooled], dim=-1)  # (batch, k + gcn_hidden)
        
        # Classify
        logits = self.classifier(combined)  # (batch, num_classes)
        
        return logits
    
    def forward_with_esm_history(self, windows_sequence):
        """
        Forward pass that also returns ESM history for visualization.
        """
        batch_size, num_windows, N, F_dim = windows_sequence.shape
        device = windows_sequence.device
        
        esm = torch.zeros(batch_size, N, N, device=device)
        esm_history = [esm.detach().cpu().numpy()]
        all_node_embeddings = []
        
        for t in range(num_windows):
            de_t = windows_sequence[:, t, :, :]
            adj_candidate = self.compute_candidate_adjacency(de_t)
            esm = self.edge_gru(esm, adj_candidate)
            node_emb = self.gcn(de_t, esm)
            all_node_embeddings.append(node_emb)
            esm_history.append(esm.detach().cpu().numpy())
        
        spectral_features = self.spectral_readout(esm)
        all_node_embeddings = torch.stack(all_node_embeddings, dim=1)
        graph_embeddings = all_node_embeddings.mean(dim=2)
        attn_weights = self.temporal_attention(graph_embeddings)
        pooled = (graph_embeddings * attn_weights).sum(dim=1)
        combined = torch.cat([spectral_features, pooled], dim=-1)
        logits = self.classifier(combined)
        
        return logits, esm_history