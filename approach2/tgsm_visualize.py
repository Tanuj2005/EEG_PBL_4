"""
Visualization tools for TGSM analysis:
- ESM evolution over time
- Graph spectral features
- Electrode connectivity patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os

from tgsm_model import TGSM
from tgsm_data_loader import NUM_CHANNELS

# DREAMER uses Emotiv EPOC with 14 channels
CHANNEL_NAMES = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# Approximate 2D positions for EPOC channels (for topographic plots)
CHANNEL_POS_2D = {
    'AF3': (-0.3, 0.9), 'AF4': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.3, 0.6), 'F4': (0.3, 0.6), 'F8': (0.7, 0.6),
    'FC5': (-0.6, 0.3), 'FC6': (0.6, 0.3),
    'T7': (-0.9, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.6, -0.4), 'P8': (0.6, -0.4),
    'O1': (-0.3, -0.7), 'O2': (0.3, -0.7)
}


def plot_esm_evolution(esm_history, save_path=None, num_steps=8):
    """
    Visualize how the Edge State Matrix evolves over time.
    
    Args:
        esm_history: list of (1, N, N) numpy arrays (from forward_with_esm_history)
        save_path: path to save figure
        num_steps: number of time steps to show
    """
    total_steps = len(esm_history)
    step_indices = np.linspace(0, total_steps - 1, num_steps, dtype=int)
    
    fig, axes = plt.subplots(2, num_steps // 2, figsize=(4 * (num_steps // 2), 8))
    axes = axes.flatten()
    
    vmin = min(esm_history[i][0].min() for i in step_indices)
    vmax = max(esm_history[i][0].max() for i in step_indices)
    
    for plot_idx, step_idx in enumerate(step_indices):
        ax = axes[plot_idx]
        esm = esm_history[step_idx][0]  # Take first batch item
        
        im = ax.imshow(esm, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f't={step_idx}', fontsize=10)
        ax.set_xticks(range(NUM_CHANNELS))
        ax.set_yticks(range(NUM_CHANNELS))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)
    
    plt.suptitle('Edge State Matrix (ESM) Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ESM evolution plot to {save_path}")
    plt.show()


def plot_graph_connectivity(esm, threshold=0.3, save_path=None, title='Brain Connectivity Graph'):
    """
    Plot the brain connectivity graph from an ESM.
    
    Args:
        esm: (N, N) numpy array
        threshold: minimum edge weight to display
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    
    # Nose marker
    ax.plot([0], [1.05], 'k^', markersize=10)
    
    # Draw edges
    esm_sym = (esm + esm.T) / 2
    max_weight = np.abs(esm_sym).max() + 1e-8
    
    for i in range(NUM_CHANNELS):
        for j in range(i + 1, NUM_CHANNELS):
            weight = esm_sym[i, j]
            if abs(weight) > threshold:
                pos_i = CHANNEL_POS_2D[CHANNEL_NAMES[i]]
                pos_j = CHANNEL_POS_2D[CHANNEL_NAMES[j]]
                
                alpha = min(abs(weight) / max_weight, 1.0)
                color = 'red' if weight > 0 else 'blue'
                linewidth = 1 + 3 * abs(weight) / max_weight
                
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                        color=color, alpha=alpha, linewidth=linewidth)
    
    # Draw nodes
    for ch_name in CHANNEL_NAMES:
        pos = CHANNEL_POS_2D[ch_name]
        ax.plot(pos[0], pos[1], 'ko', markersize=12, zorder=5)
        ax.text(pos[0], pos[1] + 0.08, ch_name, ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved connectivity plot to {save_path}")
    plt.show()


def plot_spectral_features(esm, save_path=None):
    """
    Plot the eigenvalue spectrum of the ESM.
    """
    esm_sym = (esm + esm.T) / 2
    eigenvalues = np.linalg.eigvalsh(esm_sym)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Eigenvalue spectrum
    axes[0].bar(range(len(eigenvalues)), sorted(eigenvalues, reverse=True), color='steelblue')
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('ESM Eigenvalue Spectrum')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # ESM heatmap
    im = axes[1].imshow(esm_sym, cmap='RdBu_r', aspect='equal')
    axes[1].set_xticks(range(NUM_CHANNELS))
    axes[1].set_yticks(range(NUM_CHANNELS))
    axes[1].set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
    axes[1].set_yticklabels(CHANNEL_NAMES, fontsize=7)
    axes[1].set_title('Final ESM (Symmetrized)')
    plt.colorbar(im, ax=axes[1])
    
    plt.suptitle('Graph Spectral Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectral plot to {save_path}")
    plt.show()


def visualize_trial(model, trial_windows, device='cpu', save_dir='figures'):
    """
    Full visualization pipeline for a single trial.
    
    Args:
        model: trained TGSM model
        trial_windows: (1, T, 14, 4) tensor
        device: torch device
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model.to(device)
    trial_windows = trial_windows.to(device)
    
    with torch.no_grad():
        logits, esm_history = model.forward_with_esm_history(trial_windows)
    
    pred = logits.argmax(dim=-1).item()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    print(f"Prediction: {'High' if pred == 1 else 'Low'} (probs: Low={probs[0]:.3f}, High={probs[1]:.3f})")
    
    # Plot ESM evolution
    plot_esm_evolution(esm_history, save_path=os.path.join(save_dir, 'esm_evolution.png'))
    
    # Plot final connectivity
    final_esm = esm_history[-1][0]
    plot_graph_connectivity(final_esm, threshold=0.2,
                            save_path=os.path.join(save_dir, 'connectivity.png'),
                            title=f'Final Connectivity (Pred: {"High" if pred == 1 else "Low"})')
    
    # Plot spectral features
    plot_spectral_features(final_esm, save_path=os.path.join(save_dir, 'spectral.png'))


if __name__ == '__main__':
    # Demo with random data
    print("Running visualization demo with random data...")
    
    model = TGSM(num_channels=14, num_bands=4, gcn_hidden=32, num_eigenvalues=5)
    
    # Simulate a trial: 60 windows, 14 channels, 4 frequency bands
    dummy_trial = torch.randn(1, 60, 14, 4)
    
    visualize_trial(model, dummy_trial, device='cpu', save_dir='figures')