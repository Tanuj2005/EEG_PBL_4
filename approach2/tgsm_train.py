"""
Training and evaluation pipeline for TGSM on DREAMER dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from tgsm_model import TGSM
from tgsm_data_loader import load_dreamer_dataset, create_synthetic_dreamer
from tgsm_dataset import DREAMERTrialDataset, collate_variable_length, create_kfold_splits, create_loso_splits


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_windows_per_step=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        windows = batch['windows'].to(device)   # (B, T, 14, 4)
        labels = batch['label'].to(device)       # (B,)
        lengths = batch['length']                 # (B,)
        
        # Optionally truncate very long sequences to save memory
        if max_windows_per_step is not None:
            max_len = min(windows.shape[1], max_windows_per_step)
            windows = windows[:, :max_len]
        
        optimizer.zero_grad()
        logits = model(windows)  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, max_windows_per_step=None):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        windows = batch['windows'].to(device)
        labels = batch['label'].to(device)
        
        if max_windows_per_step is not None:
            max_len = min(windows.shape[1], max_windows_per_step)
            windows = windows[:, :max_len]
        
        logits = model(windows)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def run_experiment(
    trials,
    target='valence',
    cv_method='kfold',
    n_splits=5,
    num_epochs=50,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    gcn_hidden=32,
    num_eigenvalues=5,
    dropout=0.3,
    max_windows=120,
    max_windows_per_step=None,
    device='auto',
    results_dir='results'
):
    """
    Run full cross-validation experiment.
    
    Args:
        trials: list of Trial namedtuples
        target: 'valence', 'arousal', or 'dominance'
        cv_method: 'kfold' or 'loso'
        n_splits: number of folds for k-fold
        num_epochs: training epochs per fold
        batch_size: batch size
        lr: learning rate
        weight_decay: L2 regularization
        gcn_hidden: GCN hidden dimension
        num_eigenvalues: number of eigenvalues for spectral readout
        dropout: dropout rate
        max_windows: max sequence length (truncate/pad)
        max_windows_per_step: truncate during training to save memory
        device: 'auto', 'cuda', or 'cpu'
        results_dir: directory to save results
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create splits
    if cv_method == 'kfold':
        splits = create_kfold_splits(trials, n_splits=n_splits)
    elif cv_method == 'loso':
        splits = create_loso_splits(trials)
        n_splits = len(splits)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")
    
    print(f"\nExperiment: target={target}, cv={cv_method}, {n_splits} folds")
    print(f"Hyperparams: epochs={num_epochs}, batch={batch_size}, lr={lr}, "
          f"gcn_hidden={gcn_hidden}, eigenvalues={num_eigenvalues}, dropout={dropout}")
    
    fold_results = []
    all_preds_combined = []
    all_labels_combined = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"  Train: {len(train_idx)} trials, Test: {len(test_idx)} trials")
        
        # Create datasets
        train_trials = [trials[i] for i in train_idx]
        test_trials = [trials[i] for i in test_idx]
        
        train_dataset = DREAMERTrialDataset(train_trials, target=target, max_windows=max_windows)
        test_dataset = DREAMERTrialDataset(test_trials, target=target, max_windows=max_windows)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_variable_length, num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_variable_length, num_workers=0, pin_memory=True
        )
        
        # Create model
        model = TGSM(
            num_channels=14,
            num_bands=4,
            gcn_hidden=gcn_hidden,
            num_eigenvalues=num_eigenvalues,
            num_classes=2,
            dropout=dropout
        ).to(device)
        
        # Class weights for imbalanced data
        label_counts = np.bincount(train_dataset.labels, minlength=2)
        if label_counts.min() > 0:
            class_weights = torch.FloatTensor(
                len(label_counts) / (label_counts * len(label_counts))
            ).to(device)
        else:
            class_weights = torch.ones(2).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
        
        best_test_acc = 0
        best_test_f1 = 0
        best_epoch = 0
        patience = 15
        no_improve = 0
        
        for epoch in range(num_epochs):
            t0 = time.time()
            
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, max_windows_per_step
            )
            
            test_loss, test_acc, test_f1, preds, labels = evaluate(
                model, test_loader, criterion, device, max_windows_per_step
            )
            
            scheduler.step()
            
            elapsed = time.time() - t0
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_epoch = epoch
                best_preds = preds.copy()
                best_labels = labels.copy()
                no_improve = 0
                # Save best model for this fold
                torch.save(model.state_dict(),
                           os.path.join(results_dir, f'tgsm_best_fold{fold_idx}.pt'))
            else:
                no_improve += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{num_epochs} "
                      f"[{elapsed:.1f}s] "
                      f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
                      f"Test: loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} "
                      f"{'*' if no_improve == 0 else ''}")
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n  Best: epoch={best_epoch+1}, acc={best_test_acc:.4f}, f1={best_test_f1:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(best_labels, best_preds)}")
        
        fold_results.append({
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'accuracy': best_test_acc,
            'f1': best_test_f1,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        all_preds_combined.extend(best_preds)
        all_labels_combined.extend(best_labels)
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1'] for r in fold_results]
    
    results = {
        'target': target,
        'cv_method': cv_method,
        'n_splits': n_splits,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'fold_results': fold_results,
        'hyperparams': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'gcn_hidden': gcn_hidden,
            'num_eigenvalues': num_eigenvalues,
            'dropout': dropout,
            'max_windows': max_windows
        },
        'model_params': sum(p.numel() for p in TGSM(
            gcn_hidden=gcn_hidden,
            num_eigenvalues=num_eigenvalues,
            dropout=dropout
        ).parameters() if p.requires_grad),
        'classification_report': classification_report(
            all_labels_combined, all_preds_combined, target_names=['Low', 'High'], output_dict=True
        ),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {target.upper()} ({cv_method})")
    print(f"  Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"  Model params: {results['model_params']}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels_combined, all_preds_combined, target_names=['Low', 'High']))
    
    # Save results
    results_path = os.path.join(results_dir, f'tgsm_results_{target}_{cv_method}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TGSM for DREAMER EEG Emotion Recognition')
    parser.add_argument('--data_path', type=str, default='DREAMER.mat',
                        help='Path to DREAMER.mat file')
    parser.add_argument('--target', type=str, default='valence',
                        choices=['valence', 'arousal', 'dominance'])
    parser.add_argument('--cv', type=str, default='kfold',
                        choices=['kfold', 'loso'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gcn_hidden', type=int, default=32)
    parser.add_argument('--eigenvalues', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_windows', type=int, default=120)
    parser.add_argument('--window_size', type=float, default=1.0,
                        help='Window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Window overlap fraction')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--results_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Load data
    if args.synthetic:
        trials = create_synthetic_dreamer(
            num_subjects=23, num_videos=18, trial_length_sec=60,
            window_size_sec=args.window_size, overlap=args.overlap
        )
    else:
        if not os.path.exists(args.data_path):
            print(f"DREAMER.mat not found at {args.data_path}")
            print("Using synthetic data instead. Use --data_path to specify location.")
            trials = create_synthetic_dreamer(
                num_subjects=23, num_videos=18, trial_length_sec=60,
                window_size_sec=args.window_size, overlap=args.overlap
            )
        else:
            trials = load_dreamer_dataset(
                args.data_path,
                window_size_sec=args.window_size,
                overlap=args.overlap
            )
    
    # Run experiment
    results = run_experiment(
        trials=trials,
        target=args.target,
        cv_method=args.cv,
        n_splits=args.folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gcn_hidden=args.gcn_hidden,
        num_eigenvalues=args.eigenvalues,
        dropout=args.dropout,
        max_windows=args.max_windows,
        results_dir=args.results_dir
    )