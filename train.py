"""
Training utilities: train loop, early stopping, metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
import config
from model import EmotionClassifier, create_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = config.PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, val_score: float, model: nn.Module):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def create_dataloader(X: np.ndarray, y: np.ndarray,
                      batch_size: int = config.BATCH_SIZE,
                      shuffle: bool = True) -> DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0, drop_last=False)


def train_one_epoch(model: EmotionClassifier, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: EmotionClassifier, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Dict:
    """
    Evaluate model.

    Returns:
        Dict with loss, accuracy, f1, predictions, labels
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
    }


def train_model(model: EmotionClassifier,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int = config.NUM_EPOCHS,
                lr: float = config.LEARNING_RATE,
                weight_decay: float = config.WEIGHT_DECAY) -> Dict:
    """
    Full training loop with early stopping.

    Returns:
        Dict with training history and best metrics
    """
    model = model.to(device)

    # Class-balanced loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    print(f"\n{'='*60}")
    print(f"Training: {model.count_parameters():,} parameters, device={device}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                  f"F1: {val_metrics['f1']:.4f} | {elapsed:.1f}s")

        # Early stopping
        early_stopping(val_metrics['accuracy'], model)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}. "
                  f"Best val acc: {early_stopping.best_score:.4f}")
            break

    # Load best model
    early_stopping.load_best(model)

    # Final evaluation
    final_metrics = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal Val Accuracy: {final_metrics['accuracy']:.4f}, "
          f"F1: {final_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")

    return {
        'history': history,
        'best_val_acc': early_stopping.best_score,
        'final_metrics': final_metrics,
    }