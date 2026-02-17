"""
Load and parse the DREAMER.mat dataset.
DREAMER structure:
  DREAMER.Data{subject}.EEG.stimuli{video} → [samples x 14]
  DREAMER.Data{subject}.EEG.baseline{video} → [samples x 14]
  DREAMER.Data{subject}.ScoreValence(video)
  DREAMER.Data{subject}.ScoreArousal(video)
  DREAMER.Data{subject}.ScoreDominance(video)
"""

import numpy as np
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import os
import config


def load_dreamer_mat(mat_path: str = None) -> dict:
    """Load the DREAMER.mat file."""
    if mat_path is None:
        mat_path = config.DREAMER_MAT_PATH

    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"DREAMER.mat not found at {mat_path}. "
            f"Please download it and place it in {config.DATA_DIR}/"
        )

    print(f"Loading DREAMER.mat from {mat_path}...")
    mat = sio.loadmat(mat_path, simplify_cells=True)
    return mat


def extract_subject_data(mat: dict) -> List[Dict]:
    """
    Extract per-subject, per-trial EEG data and labels.

    Returns:
        List of dicts, each containing:
            - 'subject': int
            - 'trial': int
            - 'eeg_stimuli': np.ndarray [samples x 14]
            - 'eeg_baseline': np.ndarray [samples x 14]
            - 'valence': int (1-5)
            - 'arousal': int (1-5)
            - 'dominance': int (1-5)
    """
    dreamer_data = mat['DREAMER']['Data']
    all_trials = []

    num_subjects = len(dreamer_data)
    print(f"Found {num_subjects} subjects in DREAMER dataset.")

    for subj_idx in range(num_subjects):
        subj = dreamer_data[subj_idx]

        eeg_stimuli_all = subj['EEG']['stimuli']
        eeg_baseline_all = subj['EEG']['baseline']
        valence_scores = subj['ScoreValence']
        arousal_scores = subj['ScoreArousal']
        dominance_scores = subj['ScoreDominance']

        num_trials = len(eeg_stimuli_all)

        for trial_idx in range(num_trials):
            trial_info = {
                'subject': subj_idx,
                'trial': trial_idx,
                'eeg_stimuli': np.array(eeg_stimuli_all[trial_idx], dtype=np.float32),
                'eeg_baseline': np.array(eeg_baseline_all[trial_idx], dtype=np.float32),
                'valence': int(valence_scores[trial_idx]) if np.ndim(valence_scores) > 0 else int(valence_scores),
                'arousal': int(arousal_scores[trial_idx]) if np.ndim(arousal_scores) > 0 else int(arousal_scores),
                'dominance': int(dominance_scores[trial_idx]) if np.ndim(dominance_scores) > 0 else int(dominance_scores),
            }
            all_trials.append(trial_info)

    print(f"Extracted {len(all_trials)} total trials.")
    return all_trials


def binarize_labels(score: int, threshold: float = 3.0) -> int:
    """Convert 1-5 score to binary: 0 (low, <=3) or 1 (high, >3)."""
    return 1 if score > threshold else 0


def get_subject_ids(trials: List[Dict]) -> np.ndarray:
    """Get unique subject IDs."""
    return np.unique([t['subject'] for t in trials])