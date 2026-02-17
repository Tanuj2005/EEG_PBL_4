"""
Main entry point for TGSM experiments on DREAMER.
Runs valence, arousal, and dominance with both k-fold and LOSO.
"""

import os
import json
import torch
import argparse
from tgsm_data_loader import load_dreamer_dataset, create_synthetic_dreamer
from tgsm_train import run_experiment


def main():
    parser = argparse.ArgumentParser(description='TGSM — Temporal Graph State Machine for EEG')
    parser.add_argument('--data_path', type=str, default='DREAMER.mat')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--targets', nargs='+', default=['valence', 'arousal', 'dominance'])
    parser.add_argument('--cv_methods', nargs='+', default=['kfold', 'loso'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gcn_hidden', type=int, default=32)
    parser.add_argument('--eigenvalues', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_windows', type=int, default=120)
    parser.add_argument('--window_size', type=float, default=1.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--results_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Load data once
    if args.synthetic or not os.path.exists(args.data_path):
        if not args.synthetic:
            print(f"WARNING: {args.data_path} not found — falling back to synthetic data")
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
    
    # Run all experiments
    all_results = {}
    
    for target in args.targets:
        for cv_method in args.cv_methods:
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENT: {target.upper()} — {cv_method.upper()}")
            print(f"{'#'*70}")
            
            result = run_experiment(
                trials=trials,
                target=target,
                cv_method=cv_method,
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
            
            key = f"{target}_{cv_method}"
            all_results[key] = {
                'accuracy': f"{result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                'f1': f"{result['mean_f1']:.4f} ± {result['std_f1']:.4f}",
                'params': result['model_params']
            }
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Accuracy':<20} {'F1 Score':<20} {'Params':<10}")
    print(f"{'-'*75}")
    
    for key, res in all_results.items():
        print(f"{key:<25} {res['accuracy']:<20} {res['f1']:<20} {res['params']:<10}")
    
    # Save summary
    summary_path = os.path.join(args.results_dir, 'tgsm_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()