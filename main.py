"""
Main entry point for DREAMER EEG Emotion Recognition.

Usage:
    python main.py                          # Full pipeline
    python main.py --target valence         # Specific target
    python main.py --target arousal --cv loso  # LOSO for arousal
    python main.py --test-model             # Quick model test (no data needed)
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import config


def set_seed(seed: int = config.SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model_only():
    """Quick model architecture test without data."""
    from model import create_model

    print("=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)

    model = create_model(num_classes=2)
    print(f"\n{model}\n")
    print(f"Total parameters: {model.count_parameters():,}")

    # Count per-module
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params:,} params")

    # Forward pass test
    batch_sizes = [1, 8, 32, 64]
    for bs in batch_sizes:
        x = torch.randn(bs, config.NUM_PAIRS, config.FEATURES_PER_PAIR)
        out = model(x)
        print(f"  Batch {bs}: input {x.shape} → output {out.shape}")

    # Adjacency matrix
    from model import build_adjacency_matrix
    adj = build_adjacency_matrix()
    print(f"\nAdjacency matrix ({adj.shape}):")
    pair_names = ["AF3-AF4", "F7-F8", "F3-F4", "FC5-FC6", "T7-T8", "P7-P8", "O1-O2"]
    print(f"{'':>10}", end="")
    for name in pair_names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(pair_names):
        print(f"{name:>10}", end="")
        for j in range(len(pair_names)):
            val = adj[i, j].item()
            if val > 0.01:
                print(f"{'%.2f' % val:>10}", end="")
            else:
                print(f"{'·':>10}", end="")
        print()

    print("\n✓ Model test passed!")


def run_pipeline(target: str = "valence", cv_method: str = "kfold"):
    """Run the full pipeline."""
    from data_loader import load_dreamer_mat, extract_subject_data
    from features import build_dataset
    from evaluate import subject_independent_cv, loso_cv

    set_seed()

    print("=" * 60)
    print(f"DREAMER EEG EMOTION RECOGNITION")
    print(f"Target: {target.upper()} | CV: {cv_method.upper()}")
    print(f"Device: {config.DEVICE if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # --- Step 1: Load Data ---
    print("\n[1/4] Loading DREAMER dataset...")
    t0 = time.time()
    mat = load_dreamer_mat()
    trials = extract_subject_data(mat)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Step 2: Extract Features ---
    print(f"\n[2/4] Extracting features...")
    t0 = time.time()
    X, y, subjects = build_dataset(trials, target=target)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {np.bincount(y)}")
    print(f"  Subjects: {np.unique(subjects)}")

    # --- Step 3: Save preprocessed data ---
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cache_path = os.path.join(config.RESULTS_DIR, f"features_{target}.npz")
    np.savez_compressed(cache_path, X=X, y=y, subjects=subjects)
    print(f"  Cached features to {cache_path}")

    # --- Step 4: Cross-Validation ---
    print(f"\n[3/4] Running {cv_method} cross-validation...")
    t0 = time.time()

    if cv_method == "loso":
        results = loso_cv(X, y, subjects, target_name=target)
    else:
        results = subject_independent_cv(X, y, subjects, target_name=target)

    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Step 5: Save results ---
    print(f"\n[4/4] Saving results...")
    results_summary = {
        'target': target,
        'cv_method': cv_method,
        'mean_accuracy': float(results['mean_acc']),
        'std_accuracy': float(results['std_acc']),
        'mean_f1': float(results['mean_f1']),
        'std_f1': float(results['std_f1']),
        'num_samples': int(len(y)),
        'num_features': list(X.shape),
        'model_params': None,  # filled below
    }

    from model import create_model
    temp_model = create_model(num_classes=config.NUM_CLASSES_VALENCE)
    results_summary['model_params'] = temp_model.count_parameters()

    results_path = os.path.join(config.RESULTS_DIR, f"results_{target}_{cv_method}.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Results saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {target.upper()}")
    print(f"  Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"  F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"  Model Size: {results_summary['model_params']:,} parameters")
    print(f"{'='*60}")

    return results


def run_all_targets():
    """Run pipeline for all three emotion dimensions."""
    all_results = {}

    for target in ['valence', 'arousal', 'dominance']:
        print(f"\n{'#'*60}")
        print(f"  RUNNING: {target.upper()}")
        print(f"{'#'*60}\n")
        results = run_pipeline(target=target, cv_method="kfold")
        all_results[target] = {
            'accuracy': f"{results['mean_acc']:.4f} ± {results['std_acc']:.4f}",
            'f1': f"{results['mean_f1']:.4f} ± {results['std_f1']:.4f}",
        }

    print(f"\n{'='*60}")
    print("SUMMARY - ALL TARGETS")
    print(f"{'='*60}")
    for target, metrics in all_results.items():
        print(f"  {target.upper():>12}: Acc={metrics['accuracy']}, F1={metrics['f1']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="DREAMER EEG Emotion Recognition with Hemispheric Pair Compression"
    )
    parser.add_argument('--target', type=str, default='valence',
                        choices=['valence', 'arousal', 'dominance', 'all'],
                        help='Emotion dimension to classify')
    parser.add_argument('--cv', type=str, default='kfold',
                        choices=['kfold', 'loso'],
                        help='Cross-validation method')
    parser.add_argument('--test-model', action='store_true',
                        help='Test model architecture only (no data needed)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached features if available')

    args = parser.parse_args()

    set_seed()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if args.test_model:
        test_model_only()
        return

    if args.target == 'all':
        run_all_targets()
    else:
        # Check for cached features
        if args.use_cache:
            cache_path = os.path.join(config.RESULTS_DIR, f"features_{args.target}.npz")
            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_path}...")
                data = np.load(cache_path)
                X, y, subjects = data['X'], data['y'], data['subjects']

                from evaluate import subject_independent_cv, loso_cv
                if args.cv == 'loso':
                    loso_cv(X, y, subjects, target_name=args.target)
                else:
                    subject_independent_cv(X, y, subjects, target_name=args.target)
                return

        run_pipeline(target=args.target, cv_method=args.cv)


if __name__ == "__main__":
    main()