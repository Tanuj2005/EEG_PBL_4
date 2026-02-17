"""
Configuration for DREAMER EEG Emotion Recognition.
DREAMER: 14 EEG channels (Emotiv EPOC), 23 subjects,
Valence/Arousal/Dominance labels (1-5 scale).
"""

import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DREAMER_MAT_PATH = "DREAMER.mat"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# --- DREAMER Dataset Info ---
NUM_SUBJECTS = 23
NUM_VIDEOS = 18
SAMPLING_RATE = 128  # Hz
NUM_CHANNELS = 14

# 14 Emotiv EPOC channel names (in order)
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# --- Hemispheric Pairs (Left - Right symmetric electrodes) ---
# DREAMER uses 14-channel Emotiv EPOC
# Pairs: (left_index, right_index)
HEMISPHERE_PAIRS = [
    (0, 13),   # AF3 - AF4
    (1, 12),   # F7  - F8
    (2, 11),   # F3  - F4
    (3, 10),   # FC5 - FC6
    (4, 9),    # T7  - T8
    (5, 8),    # P7  - P8
    (6, 7),    # O1  - O2
]
NUM_PAIRS = len(HEMISPHERE_PAIRS)  # 7

# --- Frequency Bands ---
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)

# --- Windowing ---
WINDOW_SIZE_SEC = 2        # seconds per segment
WINDOW_SIZE = WINDOW_SIZE_SEC * SAMPLING_RATE  # 256 samples
OVERLAP_RATIO = 0.5        # 50% overlap
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# --- Features ---
# Per pair: 3 compressions (diff, sum, ratio) x 2 bands (alpha, beta)
#   + 3 compressions x 1 DE feature = total per pair
# Actually: alpha_DE + beta_DE for each channel in pair → then compress
FEATURES_PER_PAIR = 6  # diff_alpha, sum_alpha, ratio_alpha, diff_beta, sum_beta, ratio_beta
# Plus raw DE features: 2 per pair (left_de, right_de) x 2 bands = 4 per pair
# We'll use: 3 compress * 2 bands = 6 features per pair
TOTAL_FEATURES = NUM_PAIRS * FEATURES_PER_PAIR  # 7 * 6 = 42

# --- Model ---
TEMPORAL_CHANNELS = 16
TEMPORAL_KERNEL = 5
GRAPH_HIDDEN = 32
ATTENTION_REDUCTION = 4
NUM_CLASSES_VALENCE = 2  # Binary: Low (1-3) / High (4-5)
NUM_CLASSES_AROUSAL = 2
NUM_CLASSES_DOMINANCE = 2

# --- Training ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 15  # Early stopping
DEVICE = "cuda"  # or "cpu"
SEED = 42

# --- Cross-validation ---
NUM_FOLDS = 5  # For subject-independent eval