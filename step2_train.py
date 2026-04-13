"""
Step 2: LOSO Training Pipeline
===============================
Architecture-neutral training loop.  A default model (Compact EEGNet) is
included but any nn.Module with the same forward signature can be plugged in.

Usage
-----
    python step2_train.py                  # uses default EEGNet
    python step2_train.py --epochs 50      # override epochs

The script loads LOSO splits produced by step1_preprocess.py.
Trained models are saved to  models/model_subject_<id>.pt
Results are saved to         results/training_results.pkl
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ─────────────────────────── Paths & seeds ───────────────────────────────────
PROCESSED_DIR = Path("processed")
MODELS_DIR    = Path("models");   MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR   = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDED MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  WHY Compact EEGNet?
#  ───────────────────
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  Criterion         EEGNet          Conformer      ShallowConvNet    │
#  │  ─────────────────────────────────────────────────────────────────  │
#  │  Params            ~2 k            ~300 k         ~40 k             │
#  │  Training speed    ★★★★★          ★★             ★★★★             │
#  │  LOSO performance  Strong          SOTA           Good              │
#  │  IG attribution    Excellent       Good           Good              │
#  │  Interpretability  Very high       Medium         High              │
#  └──────────────────────────────────────────────────────────────────────┘
#
#  EEGNet uses depthwise-separable convolutions that naturally respect the
#  spatial (channel) × temporal structure of EEG. Because the depth-wise
#  conv operates on each EEG channel independently its Integrated Gradients
#  attributions map cleanly back to electrode importance — exactly what the
#  downstream analysis requires.  With ~2 k parameters it trains in seconds
#  per fold even on CPU, making it ideal for a 15-fold LOSO loop.
# ═══════════════════════════════════════════════════════════════════════════════

# REMOVE the entire EEGNet class
# ADD these two classes in its place:

class ShallowConvNet(nn.Module):
    """
    Schirrmeister et al. 2017 — best raw-EEG model for IG interpretability.

    Architecture:
      1. Temporal conv  : learns frequency-selective filters
      2. Spatial conv   : depthwise, learns electrode weighting per filter
      3. Square → log   : approximates power in learned bands
      4. Dropout + FC   : classification

    Input : [B, C, T]   e.g. [B, 62, 800]
    Output: [B, n_classes]

    IG gradient path: output → FC → pool → log → square → spatial → temporal → input[channel, :]
    Each input channel gets a clean, direct gradient — ideal for channel importance.
    """
    def __init__(self, n_channels: int = 62, n_times: int = 800,
                 n_classes: int = 4, n_filters: int = 40,
                 filter_time_length: int = 25, dropout: float = 0.5):
        super().__init__()

        # Step 1 — temporal convolution across time
        self.temporal_conv = nn.Conv2d(
            1, n_filters,
            kernel_size=(1, filter_time_length),
            padding=(0, filter_time_length // 2),
            bias=False,
        )

        # Step 2 — spatial (depthwise) convolution across channels
        self.spatial_conv = nn.Conv2d(
            n_filters, n_filters,
            kernel_size=(n_channels, 1),
            groups=n_filters,          # depthwise: each filter has its own spatial weight
            bias=False,
        )

        self.bn = nn.BatchNorm2d(n_filters, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        # Compute pooled size dynamically
        with torch.no_grad():
            dummy  = torch.zeros(1, 1, n_channels, n_times)
            tmp    = self.temporal_conv(dummy)        # [1, F, C, T]
            tmp    = self.spatial_conv(tmp)           # [1, F, 1, T]
            tmp    = tmp ** 2                         # square activation
            tmp    = nn.AvgPool2d(                    # temporal pooling
                kernel_size=(1, 75), stride=(1, 15)
            )(tmp)
            tmp    = torch.log(torch.clamp(tmp, min=1e-6))
            flat   = tmp.numel()

        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unsqueeze(1)                            # [B, 1, C, T]
        x = self.temporal_conv(x)                     # [B, F, C, T]
        x = self.spatial_conv(x)                      # [B, F, 1, T]
        x = self.bn(x)
        x = x ** 2                                    # square activation
        x = self.pool(x)                              # [B, F, 1, T']
        x = torch.log(torch.clamp(x, min=1e-6))      # log activation
        x = self.dropout(x)
        return self.classifier(x.flatten(1))
# ─────────────────────────── Factory helper ──────────────────────────────────
def build_default_model(n_channels, n_times, n_classes=4):
    return ShallowConvNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        X_b, y_b = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        total      += len(y_b)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        X_b, y_b = batch[0].to(device), batch[1].to(device)
        logits    = model(X_b)
        preds_all.append(logits.argmax(1).cpu())
        labels_all.append(y_b.cpu())
    preds  = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    acc    = (preds == labels).mean()
    return acc, preds, labels




# ADD this function before run_loso_training():

def normalize_test_loader(test_loader) -> DataLoader:
    """Re-normalize test subject using its own mean/std."""
    all_X   = torch.cat([b[0] for b in test_loader], dim=0)   # [N, C, T]
    all_y   = torch.cat([b[1] for b in test_loader])
    all_sid = torch.cat([b[2] for b in test_loader])

    mu    = all_X.mean(dim=(0, 2), keepdim=True)   # [1, C, 1]
    sigma = all_X.std(dim=(0, 2),  keepdim=True) + 1e-8

    all_X = (all_X - mu) / sigma

    from torch.utils.data import TensorDataset
    ds = TensorDataset(all_X, all_y, all_sid)
    return DataLoader(ds, batch_size=test_loader.batch_size, shuffle=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP   (architecture-neutral)
# ═══════════════════════════════════════════════════════════════════════════════

def run_loso_training(
    model_factory,          # callable(n_channels, n_times, n_classes) → nn.Module
    splits: list,
    epochs: int  = 30,
    lr: float    = 1e-3,
    device: str  = None,
) -> list:
    """
    Parameters
    ----------
    model_factory : factory function returning a fresh model per fold
    splits        : list of dicts from step1  {subject_id, train_loader, test_loader}
    epochs        : training epochs per fold
    lr            : Adam learning rate
    device        : 'cpu' | 'cuda' | None (auto-detect)

    Returns
    -------
    results : list of dicts, one per subject
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on device: {device}")

    # Infer input dimensions from first batch
    first_split = splits[0]
    sample_X, sample_y, *_ = next(iter(first_split["train_loader"]))
    _, n_channels, n_times = sample_X.shape
    
    # Efficiently find max class without repeatedly iterating over all data loaders
    dataset = first_split["train_loader"].dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    
    all_labels = dataset.tensors[1]
    n_classes = int(all_labels.max().item()) + 1

    print(f"Input  : channels={n_channels}, time={n_times}")
    print(f"Classes: {n_classes}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    results   = []

    for fold in splits:
        subj_id      = fold["subject_id"]
        train_loader = fold["train_loader"]
        test_loader  = fold["test_loader"]
        test_loader = normalize_test_loader(test_loader)

        set_seed(RANDOM_SEED)
        model = model_factory(n_channels, n_times, n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)


        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n── Subject {subj_id:2d}  │  trainable params: {n_params:,} ──")

        history = {"train_loss": [], "train_acc": [], "test_acc": []}
        best_acc    = 0.0
        patience    = 10
        no_improve  = 0
        model_path  = MODELS_DIR / f"model_subject_{subj_id}.pt"

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            te_acc, _, _    = evaluate(model, test_loader, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                    f"loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

            if te_acc > best_acc:
                best_acc   = te_acc
                no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch}  (best acc: {best_acc:.4f})")
                    break

        # Final evaluation on the best saved model
        model.load_state_dict(torch.load(model_path))
        final_acc, preds, labels = evaluate(model, test_loader, device)

        print(f"  Final best test acc: {final_acc:.4f} | Saved → {model_path}")

        results.append({
            "subject_id":  subj_id,
            "model_path":  str(model_path),
            "test_loader": test_loader,
            "predictions": preds,
            "labels":      labels,
            "history":     history,
            "final_acc":   final_acc,
            # Store model config so step3 can rebuild
            "model_config": {
                "n_channels": n_channels,
                "n_times":    n_times,
                "n_classes":  n_classes,
            },
        })

    # Print summary
    accs = [r["final_acc"] for r in results]
    print(f"\n{'='*50}")
    print(f"LOSO Summary  |  mean acc: {np.mean(accs):.4f}  std: {np.std(accs):.4f}")
    for r in results:
        print(f"  Subject {r['subject_id']:2d}: {r['final_acc']:.4f}")
    print(f"{'='*50}")

    # Save
    results_path = RESULTS_DIR / "training_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    return results


# ─────────────────────────── CLI entry point ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LOSO EEG Training")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    splits_path = PROCESSED_DIR / "loso_splits.pkl"
    if not splits_path.exists():
        raise FileNotFoundError(f"Run step1_preprocess.py first.  ({splits_path} not found)")

    with open(splits_path, "rb") as f:
        splits = pickle.load(f)
    print(f"Loaded {len(splits)} LOSO folds from {splits_path}")

    results = run_loso_training(
        model_factory = build_default_model,
        splits        = splits,
        epochs        = args.epochs,
        lr            = args.lr,
        device        = args.device,
    )
    return results


if __name__ == "__main__":
    main()