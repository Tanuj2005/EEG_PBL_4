"""
Evaluation strategies:
  - Subject-dependent (within-subject)
  - Subject-independent (leave-one-subject-out or k-fold)
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, LeaveOneGroupOut
import torch
import config
from features import normalize_features
from model import create_model
from train import create_dataloader, train_model


def subject_independent_cv(X: np.ndarray, y: np.ndarray,
                           subjects: np.ndarray,
                           target_name: str = "valence",
                           num_folds: int = config.NUM_FOLDS) -> Dict:
    """
    Subject-independent cross-validation.
    Groups subjects into folds; ensures no subject leakage.

    Args:
        X: [N, 7, 6]
        y: [N]
        subjects: [N]
        target_name: for logging
        num_folds: number of CV folds

    Returns:
        Dict with per-fold metrics and averages
    """
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    unique_subjects = np.unique(subjects)
    print(f"\n{'#'*60}")
    print(f"Subject-Independent CV for {target_name.upper()}")
    print(f"Subjects: {len(unique_subjects)}, Folds: {num_folds}")
    print(f"{'#'*60}")

    # Assign subjects to folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config.SEED)
    fold_results = []

    for fold_idx, (train_subj_idx, test_subj_idx) in enumerate(kf.split(unique_subjects)):
        train_subjects = set(unique_subjects[train_subj_idx])
        test_subjects = set(unique_subjects[test_subj_idx])

        train_mask = np.isin(subjects, list(train_subjects))
        test_mask = np.isin(subjects, list(test_subjects))

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")
        print(f"Train subjects: {sorted(train_subjects)}")
        print(f"Test subjects: {sorted(test_subjects)}")
        print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
        print(f"Train class dist: {np.bincount(y_train)}, "
              f"Test class dist: {np.bincount(y_test)}")

        # Normalize
        X_train_norm, X_test_norm = normalize_features(X_train, X_test)

        # Create loaders
        train_loader = create_dataloader(X_train_norm, y_train, shuffle=True)
        test_loader = create_dataloader(X_test_norm, y_test, shuffle=False)

        # Create fresh model
        model = create_model(num_classes=config.NUM_CLASSES_VALENCE)

        # Train
        result = train_model(model, train_loader, test_loader, device)
        fold_results.append(result)

    # Aggregate results
    accs = [r['best_val_acc'] for r in fold_results]
    f1s = [r['final_metrics']['f1'] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"RESULTS for {target_name.upper()} (Subject-Independent)")
    print(f"{'='*60}")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        print(f"  Fold {i+1}: Acc={a:.4f}, F1={f:.4f}")
    print(f"  Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    return {
        'fold_results': fold_results,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
    }


def loso_cv(X: np.ndarray, y: np.ndarray,
            subjects: np.ndarray,
            target_name: str = "valence") -> Dict:
    """
    Leave-One-Subject-Out cross-validation.

    Args:
        X: [N, 7, 6]
        y: [N]
        subjects: [N]
        target_name: for logging

    Returns:
        Dict with per-subject metrics and averages
    """
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    unique_subjects = np.unique(subjects)

    print(f"\n{'#'*60}")
    print(f"LOSO CV for {target_name.upper()}")
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"{'#'*60}")

    subject_results = []

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f"  Subject {test_subj}: Skipped (single class in test)")
            continue

        print(f"\n--- Subject {test_subj} (Test: {len(y_test)} samples) ---")

        # Normalize
        X_train_norm, X_test_norm = normalize_features(X_train, X_test)

        # Quick training (fewer epochs for LOSO speed)
        train_loader = create_dataloader(X_train_norm, y_train, shuffle=True)
        test_loader = create_dataloader(X_test_norm, y_test, shuffle=False)

        model = create_model(num_classes=config.NUM_CLASSES_VALENCE)
        result = train_model(model, train_loader, test_loader, device,
                             num_epochs=min(config.NUM_EPOCHS, 50))
        result['subject'] = test_subj
        subject_results.append(result)

    accs = [r['best_val_acc'] for r in subject_results]
    f1s = [r['final_metrics']['f1'] for r in subject_results]

    print(f"\n{'='*60}")
    print(f"LOSO RESULTS for {target_name.upper()}")
    print(f"{'='*60}")
    print(f"  Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    return {
        'subject_results': subject_results,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
    }