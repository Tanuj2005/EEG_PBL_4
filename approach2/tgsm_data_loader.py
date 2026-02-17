"""
DREAMER dataset loader with overlapping window segmentation and DE feature extraction.
"""

import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt
from collections import namedtuple
import os

# DREAMER: 23 subjects, 18 videos, 14 EEG channels, 128 Hz sampling rate
DREAMER_FS = 128
NUM_CHANNELS = 14
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
NUM_BANDS = len(FREQ_BANDS)

Trial = namedtuple('Trial', ['windows', 'valence', 'arousal', 'dominance', 'subject_id', 'video_id'])


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_de_features(segment, fs=DREAMER_FS):
    """
    Compute Differential Entropy (DE) features for one window.
    DE = 0.5 * ln(2 * pi * e * variance) for Gaussian-distributed band-filtered signal.
    
    Args:
        segment: (num_channels, num_samples) EEG segment
        fs: sampling frequency
    
    Returns:
        de_features: (num_channels, num_bands) DE feature matrix
    """
    num_channels = segment.shape[0]
    de_features = np.zeros((num_channels, NUM_BANDS))
    
    for i, (band_name, (low, high)) in enumerate(FREQ_BANDS.items()):
        filtered = bandpass_filter(segment, low, high, fs, order=5)
        # DE = 0.5 * log(2 * pi * e * var)
        variance = np.var(filtered, axis=-1)
        variance = np.maximum(variance, 1e-10)  # numerical stability
        de_features[:, i] = 0.5 * np.log(2 * np.pi * np.e * variance)
    
    return de_features


def segment_into_windows(eeg_data, window_size_sec=1.0, overlap=0.5, fs=DREAMER_FS):
    """
    Segment EEG trial into overlapping windows.
    
    Args:
        eeg_data: (num_channels, num_samples)
        window_size_sec: window duration in seconds
        overlap: fraction of overlap between consecutive windows
        fs: sampling frequency
    
    Returns:
        windows: list of (num_channels, window_samples) arrays
    """
    window_samples = int(window_size_sec * fs)
    step_samples = int(window_samples * (1 - overlap))
    num_samples = eeg_data.shape[1]
    
    windows = []
    start = 0
    while start + window_samples <= num_samples:
        windows.append(eeg_data[:, start:start + window_samples])
        start += step_samples
    
    return windows


def load_dreamer_dataset(mat_path, window_size_sec=1.0, overlap=0.5, binary_threshold=3.0):
    """
    Load DREAMER .mat file and extract trials with DE features.
    
    Args:
        mat_path: path to DREAMER.mat
        window_size_sec: window size in seconds
        overlap: overlap fraction
        binary_threshold: threshold for binary emotion classification (<=threshold -> 0, >threshold -> 1)
    
    Returns:
        trials: list of Trial namedtuples
    """
    print(f"Loading DREAMER dataset from {mat_path}...")
    mat_data = sio.loadmat(mat_path, simplify_cells=True)
    dreamer = mat_data['DREAMER']
    
    subjects = dreamer['Data']
    trials = []
    
    for subj_idx, subject in enumerate(subjects):
        eeg_data_list = subject['EEG']['stimuli']
        valence_scores = subject['ScoreValence']
        arousal_scores = subject['ScoreArousal']
        dominance_scores = subject['ScoreDominance']
        
        num_videos = len(eeg_data_list)
        
        for vid_idx in range(num_videos):
            eeg_raw = eeg_data_list[vid_idx]  # (num_samples, num_channels)
            
            if eeg_raw.ndim == 1:
                continue
            
            eeg_raw = eeg_raw.T  # -> (num_channels, num_samples)
            
            # Segment into overlapping windows and compute DE
            raw_windows = segment_into_windows(eeg_raw, window_size_sec, overlap)
            
            if len(raw_windows) < 2:
                continue
            
            de_windows = []
            for w in raw_windows:
                de = compute_de_features(w)
                de_windows.append(de)  # each is (14, 4)
            
            de_windows = np.array(de_windows)  # (num_windows, 14, 4)
            
            # Get labels
            v = valence_scores[vid_idx] if hasattr(valence_scores, '__len__') else valence_scores
            a = arousal_scores[vid_idx] if hasattr(arousal_scores, '__len__') else arousal_scores
            d = dominance_scores[vid_idx] if hasattr(dominance_scores, '__len__') else dominance_scores
            
            # Binary labels
            v_label = 1 if v > binary_threshold else 0
            a_label = 1 if a > binary_threshold else 0
            d_label = 1 if d > binary_threshold else 0
            
            trial = Trial(
                windows=de_windows,
                valence=v_label,
                arousal=a_label,
                dominance=d_label,
                subject_id=subj_idx,
                video_id=vid_idx
            )
            trials.append(trial)
    
    print(f"Loaded {len(trials)} trials from {len(subjects)} subjects")
    print(f"  Windows per trial: {np.mean([t.windows.shape[0] for t in trials]):.1f} avg")
    print(f"  Valence distribution: {sum(t.valence for t in trials)}/{len(trials)} positive")
    print(f"  Arousal distribution: {sum(t.arousal for t in trials)}/{len(trials)} positive")
    
    return trials


def create_synthetic_dreamer(num_subjects=23, num_videos=18, trial_length_sec=60,
                              window_size_sec=1.0, overlap=0.5, fs=DREAMER_FS):
    """
    Create synthetic DREAMER-like data for testing when the real dataset isn't available.
    """
    print("Creating synthetic DREAMER-like dataset...")
    trials = []
    
    for subj_idx in range(num_subjects):
        for vid_idx in range(num_videos):
            num_samples = int(trial_length_sec * fs)
            eeg_raw = np.random.randn(NUM_CHANNELS, num_samples) * 10  # µV scale
            
            raw_windows = segment_into_windows(eeg_raw, window_size_sec, overlap, fs)
            
            de_windows = []
            for w in raw_windows:
                de = compute_de_features(w, fs)
                de_windows.append(de)
            
            de_windows = np.array(de_windows)
            
            v_label = np.random.randint(0, 2)
            a_label = np.random.randint(0, 2)
            d_label = np.random.randint(0, 2)
            
            trial = Trial(
                windows=de_windows,
                valence=v_label,
                arousal=a_label,
                dominance=d_label,
                subject_id=subj_idx,
                video_id=vid_idx
            )
            trials.append(trial)
    
    print(f"Created {len(trials)} synthetic trials")
    return trials