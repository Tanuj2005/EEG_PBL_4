"""
Feature extraction:
  - Differential Entropy (DE) per channel per band
  - Hemispheric Pair Compression (Diff, Sum, Ratio)
"""

import numpy as np
from typing import Tuple, List, Dict
import config
from preprocessing import preprocess_trial
from data_loader import binarize_labels


def differential_entropy(signal: np.ndarray) -> float:
    """
    Compute Differential Entropy (DE) for a 1D signal.
    Assuming Gaussian distribution:
        DE = 0.5 * ln(2 * pi * e * variance)

    Args:
        signal: 1D array

    Returns:
        DE value (float)
    """
    var = np.var(signal)
    if var < 1e-10:
        var = 1e-10  # avoid log(0)
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def compute_de_features(segment: np.ndarray) -> np.ndarray:
    """
    Compute DE for each channel in a segment.

    Args:
        segment: [window_size x num_channels]

    Returns:
        de_values: [num_channels] array of DE values
    """
    num_channels = segment.shape[1]
    de_values = np.array([differential_entropy(segment[:, ch]) for ch in range(num_channels)],
                         dtype=np.float32)
    return de_values


def hemispheric_pair_compression(de_left: float, de_right: float,
                                  eps: float = 1e-8) -> Tuple[float, float, float]:
    """
    Compute 3 compression values for a hemispheric pair.

    Args:
        de_left: DE of left electrode
        de_right: DE of right electrode

    Returns:
        (difference, sum, ratio)
    """
    diff = de_left - de_right              # Asymmetry index
    summ = de_left + de_right              # Overall activation
    ratio = de_left / (de_right + eps)     # Relative activation
    return diff, summ, ratio


def extract_pair_features(alpha_segment: np.ndarray,
                          beta_segment: np.ndarray) -> np.ndarray:
    """
    Extract hemispheric pair compression features from alpha and beta segments.

    Args:
        alpha_segment: [window_size x 14]
        beta_segment: [window_size x 14]

    Returns:
        features: [NUM_PAIRS x FEATURES_PER_PAIR] = [7 x 6]
            For each pair: [diff_alpha, sum_alpha, ratio_alpha,
                           diff_beta, sum_beta, ratio_beta]
    """
    alpha_de = compute_de_features(alpha_segment)  # [14]
    beta_de = compute_de_features(beta_segment)    # [14]

    pair_features = []

    for left_idx, right_idx in config.HEMISPHERE_PAIRS:
        # Alpha band compression
        a_diff, a_sum, a_ratio = hemispheric_pair_compression(
            alpha_de[left_idx], alpha_de[right_idx]
        )
        # Beta band compression
        b_diff, b_sum, b_ratio = hemispheric_pair_compression(
            beta_de[left_idx], beta_de[right_idx]
        )

        pair_features.append([a_diff, a_sum, a_ratio, b_diff, b_sum, b_ratio])

    return np.array(pair_features, dtype=np.float32)  # [7, 6]


def extract_temporal_pair_features(alpha_segments: np.ndarray,
                                    beta_segments: np.ndarray) -> np.ndarray:
    """
    Extract features across all segments (temporal dimension preserved).

    Args:
        alpha_segments: [num_segments x window_size x 14]
        beta_segments: [num_segments x window_size x 14]

    Returns:
        features: [num_segments x NUM_PAIRS x FEATURES_PER_PAIR]
                 = [num_segments x 7 x 6]
    """
    num_segments = alpha_segments.shape[0]
    all_features = []

    for i in range(num_segments):
        feat = extract_pair_features(alpha_segments[i], beta_segments[i])
        all_features.append(feat)

    return np.array(all_features, dtype=np.float32)


def build_dataset(trials: List[Dict],
                  target: str = 'valence') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build complete feature dataset from trials.

    Args:
        trials: List of trial dicts from data_loader
        target: 'valence', 'arousal', or 'dominance'

    Returns:
        X: [total_segments x num_pairs x features_per_pair] = [N x 7 x 6]
        y: [total_segments] binary labels
        subjects: [total_segments] subject IDs
    """
    all_X = []
    all_y = []
    all_subjects = []

    for idx, trial in enumerate(trials):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing trial {idx + 1}/{len(trials)}...")

        try:
            alpha_segs, beta_segs = preprocess_trial(
                trial['eeg_stimuli'], trial['eeg_baseline']
            )
        except Exception as e:
            print(f"  Warning: Skipping trial {idx} (subject {trial['subject']}, "
                  f"trial {trial['trial']}): {e}")
            continue

        features = extract_temporal_pair_features(alpha_segs, beta_segs)  # [S, 7, 6]
        label = binarize_labels(trial[target])
        num_segs = features.shape[0]

        all_X.append(features)
        all_y.extend([label] * num_segs)
        all_subjects.extend([trial['subject']] * num_segs)

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)

    print(f"Dataset built: X={X.shape}, y={y.shape}, "
          f"class distribution: {np.bincount(y)}")
    return X, y, subjects


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score normalization fitted on train set.

    Args:
        X_train: [N_train x 7 x 6]
        X_test: [N_test x 7 x 6]

    Returns:
        Normalized X_train, X_test
    """
    orig_shape_train = X_train.shape
    orig_shape_test = X_test.shape

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    mean = X_train_flat.mean(axis=0, keepdims=True)
    std = X_train_flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8

    X_train_norm = ((X_train_flat - mean) / std).reshape(orig_shape_train)
    X_test_norm = ((X_test_flat - mean) / std).reshape(orig_shape_test)

    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)