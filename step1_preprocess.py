"""
Step 1: SEED-IV EEG Preprocessing & LOSO Split Creation
========================================================
Loads raw .mat files, applies bandpass filtering + per-subject z-score normalization,
segments into fixed windows, and creates Leave-One-Subject-Out DataLoaders.

Dataset layout expected (relative to script or set via SEED_IV_ROOT env var):
  eeg_raw_data/
    1/   (session 1)
      <Subject>_<Date>.mat
    2/   (session 2)
    3/   (session 3)
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

warnings.filterwarnings("ignore")

# ─────────────────────────── Configuration ───────────────────────────────────
SEED_IV_ROOT = Path(os.environ.get("SEED_IV", "SEED_IV"))
OUTPUT_DIR   = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

SFREQ        = 200          # Hz  (SEED-IV sampling rate after down-sampling in raw files)
WINDOW_SEC   = 4.0          # seconds  (matches the dataset's stated window size)
WINDOW_SAMP  = int(WINDOW_SEC * SFREQ)   # 800 samples
STEP_SAMP    = WINDOW_SAMP  # non-overlapping windows
N_CHANNELS   = 62
BATCH_SIZE   = 32
RANDOM_SEED  = 42

# Labels per session (0=neutral, 1=sad, 2=fear, 3=happy)
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

CHANNEL_NAMES = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ",
    "C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7",
    "P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6",
    "PO8","CB1","O1","OZ","O2","CB2",
]

# ─────────────────────────── Filtering ───────────────────────────────────────
def bandpass_filter(data: np.ndarray, lo=1.0, hi=50.0, fs=SFREQ) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter. data: [channels, time]"""
    sos = butter(5, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)

# ─────────────────────────── Window segmentation ─────────────────────────────
def segment(eeg: np.ndarray, label: int, subject_id: int,
            window=WINDOW_SAMP, step=STEP_SAMP):
    """
    Slice a [channels, time] array into fixed windows.
    Returns lists of (window_array, label, subject_id).
    """
    windows, labels, sids = [], [], []
    n_time = eeg.shape[1]
    start = 0
    while start + window <= n_time:
        win = normalize_window(eeg[:, start:start+window]).astype(np.float32)
        windows.append(win)
        labels.append(label)
        sids.append(subject_id)
        start += step
    return windows, labels, sids

# ─────────────────────────── Per-subject normalisation ───────────────────────
def normalize_subject(windows: list) -> list:
    """Z-score each channel across all windows of one subject."""
    stacked = np.stack(windows, axis=0)          # [N, C, T]
    mu  = stacked.mean(axis=(0, 2), keepdims=True)   # [1, C, 1]
    sigma = stacked.std(axis=(0, 2), keepdims=True) + 1e-8
    stacked = (stacked - mu) / sigma
    return [stacked[i] for i in range(len(windows))]


# ADD this function after the bandpass_filter() function

def normalize_window(window: np.ndarray) -> np.ndarray:
    """Per-channel z-score at window level. window: [C, T]"""
    mu    = window.mean(axis=-1, keepdims=True)   # [C, 1]
    sigma = window.std(axis=-1,  keepdims=True) + 1e-8
    return (window - mu) / sigma

# ─────────────────────────── Load one subject ────────────────────────────────
def load_subject(subject_idx: int, session_dirs: dict) -> tuple:
    """
    Loads all 3 sessions for subject `subject_idx` (1-based).
    Returns raw windows, labels, subject_ids before global normalization.
    """
    all_windows, all_labels, all_sids = [], [], []

    for sess_id, sess_path in session_dirs.items():
        mat_files = sorted(sess_path.glob("*.mat"), key=lambda p: int(p.stem.split('_')[0]))
        if subject_idx > len(mat_files):
            print(f"  [WARN] Session {sess_id}: only {len(mat_files)} subjects found, "
                  f"skipping subject {subject_idx}")
            continue
        mat_file = mat_files[subject_idx - 1]
        print(f"  Loading session {sess_id}: {mat_file.name}")

        data = sio.loadmat(str(mat_file))
        sess_labels = SESSION_LABELS[sess_id]

        for trial_idx in range(24):
            # Key pattern: <prefix>_eeg<trial+1>  — prefix varies per subject
            eeg_key = None
            for k in data.keys():
                if k.endswith(f"_eeg{trial_idx + 1}") or k == f"eeg{trial_idx+1}":
                    eeg_key = k
                    break
            if eeg_key is None:
                # Fallback: search keys ending with the trial number
                candidates = [k for k in data.keys()
                              if not k.startswith("__") and k.endswith(str(trial_idx + 1))]
                if candidates:
                    eeg_key = candidates[0]
            if eeg_key is None:
                print(f"  [WARN] Trial {trial_idx+1} key not found in {mat_file.name}")
                continue

            eeg = data[eeg_key]   # expected [channels, time] or [time, channels]
            if eeg.ndim != 2:
                continue
            # Ensure [channels, time]
            if eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T
            if eeg.shape[0] != N_CHANNELS:
                print(f"  [WARN] Unexpected channel count {eeg.shape[0]} in {eeg_key}")
                continue

            eeg = bandpass_filter(eeg)
            w, l, s = segment(eeg, sess_labels[trial_idx], subject_idx)
            all_windows.extend(w)
            all_labels.extend(l)
            all_sids.extend(s)

    return all_windows, all_labels, all_sids

# ─────────────────────────── Main loading routine ────────────────────────────
def load_dataset():
    eeg_root = SEED_IV_ROOT / "eeg_raw_data"
    session_dirs = {i: eeg_root / str(i) for i in [1, 2, 3]}

    for sid, p in session_dirs.items():
        if not p.exists():
            raise FileNotFoundError(f"Session directory not found: {p}")

    # Determine number of subjects from session 1
    n_subjects = len(list(session_dirs[1].glob("*.mat")))
    print(f"Found {n_subjects} subjects in session 1.")

    all_X, all_y, all_sids = [], [], []

    for subj in range(1, n_subjects + 1):
        print(f"\n── Subject {subj}/{n_subjects} ──")
        windows, labels, sids = load_subject(subj, session_dirs)
        if not windows:
            print(f"  No data for subject {subj}, skipping.")
            continue
        
        all_X.extend(windows)
        all_y.extend(labels)
        all_sids.extend(sids)

    X = np.stack(all_X, axis=0)          # [N, C, T]
    y = np.array(all_y, dtype=np.int64)
    subject_ids = np.array(all_sids, dtype=np.int64)

    return X, y, subject_ids

# ─────────────────────────── Dataset statistics ──────────────────────────────
def print_statistics(X, y, subject_ids):
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total samples   : {len(X)}")
    print(f"Tensor shape    : {X.shape}  [samples, channels, time]")
    print(f"Number of subjects: {len(np.unique(subject_ids))}")
    print(f"Emotion classes : {np.unique(y)}  (0=neutral,1=sad,2=fear,3=happy)")
    print("\nSamples per subject:")
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        counts = np.bincount(y[mask], minlength=4)
        print(f"  Subject {s:2d}: {mask.sum():5d} samples  "
              f"[neutral={counts[0]}, sad={counts[1]}, fear={counts[2]}, happy={counts[3]}]")
    print("="*50)

# ─────────────────────────── LOSO splits ─────────────────────────────────────
def create_loso_splits(X, y, subject_ids, batch_size=BATCH_SIZE):
    X_t  = torch.tensor(X,           dtype=torch.float32)
    y_t  = torch.tensor(y,           dtype=torch.long)
    sid_t = torch.tensor(subject_ids, dtype=torch.long)

    full_ds = TensorDataset(X_t, y_t, sid_t)

    splits = []
    for subj in np.unique(subject_ids):
        test_mask  = subject_ids == subj
        train_mask = ~test_mask
        
        train_indices = np.where(train_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        train_ds = Subset(full_ds, train_indices)
        test_ds  = Subset(full_ds, test_indices)

        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, generator=g,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0, pin_memory=False)

        splits.append({
            "subject_id":   int(subj),
            "train_loader": train_loader,
            "test_loader":  test_loader,
        })
        print(f"  LOSO split subject {subj:2d} | "
              f"train={train_mask.sum():5d}  test={test_mask.sum():4d}")

    return splits

# ─────────────────────────── Entry point ─────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cache_path = OUTPUT_DIR / "dataset.pkl"

    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            X, y, subject_ids = pickle.load(f)
    else:
        print("Loading & preprocessing SEED-IV dataset …")
        X, y, subject_ids = load_dataset()
        with open(cache_path, "wb") as f:
            pickle.dump((X, y, subject_ids), f)
        print(f"Dataset saved to {cache_path}")

    print_statistics(X, y, subject_ids)

    np.save(OUTPUT_DIR / "X.npy",           X)
    np.save(OUTPUT_DIR / "y.npy",           y)
    np.save(OUTPUT_DIR / "subject_ids.npy", subject_ids)

    print("\nCreating LOSO splits …")
    splits = create_loso_splits(X, y, subject_ids)

    splits_path = OUTPUT_DIR / "loso_splits.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nLOSO splits saved to {splits_path}")
    print(f"Total folds: {len(splits)}")

    return splits


if __name__ == "__main__":
    splits = main()