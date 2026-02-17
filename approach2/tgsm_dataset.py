"""
PyTorch Dataset and data utilities for TGSM training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


class DREAMERTrialDataset(Dataset):
    """
    PyTorch dataset for DREAMER trials.
    Each item is a sequence of DE feature windows + emotion label.
    """
    
    def __init__(self, trials, target='valence', max_windows=None, normalize=True):
        """
        Args:
            trials: list of Trial namedtuples
            target: 'valence', 'arousal', or 'dominance'
            max_windows: if set, truncate/pad sequences to this length
            normalize: whether to z-normalize DE features
        """
        self.target = target
        self.max_windows = max_windows
        
        # Collect all data
        self.windows_list = []
        self.labels = []
        self.subject_ids = []
        self.lengths = []
        
        for trial in trials:
            self.windows_list.append(trial.windows)  # (T, 14, 4)
            self.labels.append(getattr(trial, target))
            self.subject_ids.append(trial.subject_id)
            self.lengths.append(trial.windows.shape[0])
        
        # Determine max_windows for padding
        if self.max_windows is None:
            self.max_windows = max(self.lengths)
        
        # Normalize DE features globally
        if normalize:
            all_feats = np.concatenate([w.reshape(-1, w.shape[-1]) for w in self.windows_list])
            self.scaler = StandardScaler()
            self.scaler.fit(all_feats)
            
            for i in range(len(self.windows_list)):
                orig_shape = self.windows_list[i].shape
                flat = self.windows_list[i].reshape(-1, orig_shape[-1])
                flat = self.scaler.transform(flat)
                self.windows_list[i] = flat.reshape(orig_shape)
        
        self.labels = np.array(self.labels)
        self.subject_ids = np.array(self.subject_ids)
        
        print(f"Dataset: {len(self.labels)} trials, target={target}")
        print(f"  Max windows: {self.max_windows}")
        print(f"  Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        windows = self.windows_list[idx]  # (T, 14, 4)
        label = self.labels[idx]
        length = self.lengths[idx]
        
        T = windows.shape[0]
        
        # Pad or truncate to max_windows
        if T >= self.max_windows:
            windows = windows[:self.max_windows]
            length = self.max_windows
        else:
            pad = np.zeros((self.max_windows - T, windows.shape[1], windows.shape[2]))
            windows = np.concatenate([windows, pad], axis=0)
        
        return {
            'windows': torch.FloatTensor(windows),       # (max_windows, 14, 4)
            'label': torch.LongTensor([label]).squeeze(),  # scalar
            'length': torch.LongTensor([length]).squeeze(), # scalar
            'subject_id': self.subject_ids[idx]
        }


def collate_variable_length(batch):
    """
    Custom collate function that handles variable-length sequences.
    Already padded in __getitem__, so just stack.
    """
    windows = torch.stack([b['windows'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    lengths = torch.stack([b['length'] for b in batch])
    subject_ids = np.array([b['subject_id'] for b in batch])
    
    return {
        'windows': windows,
        'label': labels,
        'length': lengths,
        'subject_ids': subject_ids
    }


def create_kfold_splits(trials, n_splits=5, seed=42):
    """Create k-fold cross-validation splits at trial level."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(trials))
    splits = []
    for train_idx, test_idx in kf.split(indices):
        splits.append((train_idx, test_idx))
    return splits


def create_loso_splits(trials):
    """Create Leave-One-Subject-Out cross-validation splits."""
    subject_ids = np.array([t.subject_id for t in trials])
    logo = LeaveOneGroupOut()
    splits = []
    for train_idx, test_idx in logo.split(np.arange(len(trials)), groups=subject_ids):
        splits.append((train_idx, test_idx))
    return splits