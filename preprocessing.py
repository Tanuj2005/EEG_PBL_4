"""
Preprocessing: Bandpass filtering (Alpha/Beta) and segmentation.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, List
import config


def notch_filter(data: np.ndarray, fs: int = config.SAMPLING_RATE,
                 freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove power line noise."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data, axis=0)


def bandpass_filter(data: np.ndarray, low: float, high: float,
                    fs: int = config.SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        data: [samples x channels]
        low, high: band edges in Hz
        fs: sampling rate
        order: filter order

    Returns:
        Filtered data [samples x channels]
    """
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    # Clamp to valid range
    low_n = max(low_n, 0.001)
    high_n = min(high_n, 0.999)

    b, a = butter(order, [low_n, high_n], btype='band')
    filtered = filtfilt(b, a, data, axis=0)
    return filtered.astype(np.float32)


def extract_alpha_beta(eeg: np.ndarray, fs: int = config.SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract alpha and beta bands from raw EEG.

    Args:
        eeg: [samples x 14]

    Returns:
        alpha_data: [samples x 14]
        beta_data: [samples x 14]
    """
    alpha = bandpass_filter(eeg, config.ALPHA_BAND[0], config.ALPHA_BAND[1], fs)
    beta = bandpass_filter(eeg, config.BETA_BAND[0], config.BETA_BAND[1], fs)
    return alpha, beta


def baseline_correction(stimuli: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Remove baseline mean from stimuli data per channel.

    Args:
        stimuli: [samples x 14]
        baseline: [samples x 14]

    Returns:
        Corrected stimuli
    """
    baseline_mean = np.mean(baseline, axis=0, keepdims=True)
    return stimuli - baseline_mean


def segment_signal(data: np.ndarray,
                   window_size: int = config.WINDOW_SIZE,
                   step_size: int = config.STEP_SIZE) -> np.ndarray:
    """
    Segment continuous EEG into overlapping windows.

    Args:
        data: [samples x channels]
        window_size: number of samples per window
        step_size: step between windows

    Returns:
        segments: [num_segments x window_size x channels]
    """
    num_samples, num_channels = data.shape
    segments = []

    start = 0
    while start + window_size <= num_samples:
        segment = data[start:start + window_size, :]
        segments.append(segment)
        start += step_size

    if len(segments) == 0:
        # If signal too short, pad it
        padded = np.zeros((window_size, num_channels), dtype=data.dtype)
        padded[:min(num_samples, window_size), :] = data[:min(num_samples, window_size), :]
        segments.append(padded)

    return np.array(segments, dtype=np.float32)


def preprocess_trial(eeg_stimuli: np.ndarray, eeg_baseline: np.ndarray,
                     fs: int = config.SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single trial.

    Args:
        eeg_stimuli: [samples x 14]
        eeg_baseline: [samples x 14]

    Returns:
        alpha_segments: [num_segments x window_size x 14]
        beta_segments: [num_segments x window_size x 14]
    """
    # Step 1: Notch filter (50 Hz)
    eeg_stimuli = notch_filter(eeg_stimuli, fs)
    eeg_baseline = notch_filter(eeg_baseline, fs)

    # Step 2: Baseline correction
    eeg_corrected = baseline_correction(eeg_stimuli, eeg_baseline)

    # Step 3: Bandpass into alpha and beta
    alpha_data, beta_data = extract_alpha_beta(eeg_corrected, fs)

    # Step 4: Segment
    alpha_segments = segment_signal(alpha_data)
    beta_segments = segment_signal(beta_data)

    # Ensure same number of segments
    min_segs = min(len(alpha_segments), len(beta_segments))
    alpha_segments = alpha_segments[:min_segs]
    beta_segments = beta_segments[:min_segs]

    return alpha_segments, beta_segments