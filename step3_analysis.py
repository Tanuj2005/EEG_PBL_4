"""
Step 3: Channel Importance via Integrated Gradients + Cross-Subject Stability
==============================================================================
PART 1  – Integrated Gradients attributions per sample per subject
PART 2  – Subject-level aggregation → importance matrix [subjects × channels]
PART 3  – Stability metrics (variance, Spearman, Jaccard)
PART 4  – Per-emotion analysis
PART 5  – Visualisations (bar plots, heatmaps, MNE topomap)
PART 6  – Summary report (CSV + Markdown)

Requires:  captum, mne, scipy, seaborn, matplotlib, numpy, torch
"""

import pickle
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")

# ─────────────────────────── Paths ───────────────────────────────────────────
RESULTS_DIR = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR  = Path("models")
PLOTS_DIR   = Path("plots");    PLOTS_DIR.mkdir(exist_ok=True)

CHANNEL_NAMES = [
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ",
    "C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7",
    "P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6",
    "PO8","CB1","O1","OZ","O2","CB2",
]
N_CHANNELS = len(CHANNEL_NAMES)
EMOTION_MAP = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
TOP_K = 5

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL RE-IMPORT  (same EEGNet definition so weights load correctly)
# ═══════════════════════════════════════════════════════════════════════════════
# REMOVE the entire EEGNet class
# ADD these two classes in its place:

# REMOVE both PatchEmbedding and EEGConformer classes entirely
# ADD this in their place:

def normalize_test_loader(test_loader) -> DataLoader:
    """Re-normalize test subject using its own mean/std."""
    from torch.utils.data import TensorDataset, DataLoader
    all_X   = torch.cat([b[0] for b in test_loader], dim=0)
    all_y   = torch.cat([b[1] for b in test_loader])
    all_sid = torch.cat([b[2] for b in test_loader])

    mu    = all_X.mean(dim=(0, 2), keepdim=True)
    sigma = all_X.std(dim=(0, 2),  keepdim=True) + 1e-8
    all_X = (all_X - mu) / sigma

    ds = TensorDataset(all_X, all_y, all_sid)
    return DataLoader(ds, batch_size=test_loader.batch_size, shuffle=False)

class ShallowConvNet(nn.Module):
    def __init__(self, n_channels: int = 62, n_times: int = 800,
                 n_classes: int = 4, n_filters: int = 40,
                 filter_time_length: int = 25, dropout: float = 0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            1, n_filters,
            kernel_size=(1, filter_time_length),
            padding=(0, filter_time_length // 2),
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            n_filters, n_filters,
            kernel_size=(n_channels, 1),
            groups=n_filters,
            bias=False,
        )
        self.bn      = nn.BatchNorm2d(n_filters, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            tmp   = self.temporal_conv(dummy)
            tmp   = self.spatial_conv(tmp)
            tmp   = tmp ** 2
            tmp   = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))(tmp)
            tmp   = torch.log(torch.clamp(tmp, min=1e-6))
            flat  = tmp.numel()

        self.pool       = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return self.classifier(x.flatten(1))


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 – INTEGRATED GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ig(model: nn.Module, x: torch.Tensor, target: int,
               n_steps: int = 50, device: str = "cpu") -> np.ndarray:
    """
    Integrated Gradients for a single sample.

    x      : [C, T]  float32 tensor
    target : predicted class index
    returns: attribution [C, T]
    """
    x = x.to(device).unsqueeze(0)          # [1, C, T]
    baseline = torch.zeros_like(x)
    x.requires_grad_(False)

    alphas = torch.linspace(0, 1, n_steps, device=device)
    grads  = []
    for alpha in alphas:
        inp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        out = model(inp)
        score = out[0, target]
        score.backward()
        grads.append(inp.grad.squeeze(0).cpu().detach().numpy())

    grads  = np.stack(grads, axis=0)              # [steps, C, T]
    ig     = (x - baseline).squeeze(0).cpu().numpy() * grads.mean(axis=0)
    return ig                                      # [C, T]


def compute_subject_attributions(model, test_loader, device="cpu"):
    """
    Returns
    -------
    attr_list  : list of [C, T] arrays, one per sample
    pred_list  : list of predicted class indices
    label_list : list of ground-truth labels
    """
    model.eval()
    attr_list, pred_list, label_list = [], [], []

    for batch in test_loader:
        X_b, y_b = batch[0], batch[1]
        with torch.no_grad():
            logits = model(X_b.to(device))
            preds  = logits.argmax(1).cpu().numpy()

        for i in range(len(X_b)):
            x_i    = X_b[i]          # [C, T]
            pred_i = int(preds[i])
            ig     = compute_ig(model, x_i, pred_i, device=device)
            attr_list.append(ig)
            pred_list.append(pred_i)
            label_list.append(int(y_b[i]))

    return attr_list, pred_list, label_list


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 – SUBJECT-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def attribution_to_channel_importance(attrs: list) -> np.ndarray:
    """
    attrs : list of [C, T] arrays
    returns: [C] mean absolute importance
    """
    stacked = np.stack([np.abs(a).mean(axis=-1) for a in attrs], axis=0)  # [N, C]
    return stacked.mean(axis=0)                                            # [C]


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 – STABILITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def spearman_matrix(importance_matrix: np.ndarray) -> np.ndarray:
    """importance_matrix: [S, C] → correlation matrix [S, S]"""
    S = importance_matrix.shape[0]
    corr_mat = np.eye(S)
    for i, j in combinations(range(S), 2):
        r, _ = spearmanr(importance_matrix[i], importance_matrix[j])
        corr_mat[i, j] = corr_mat[j, i] = r
    return corr_mat


def jaccard_topk(importance_matrix: np.ndarray, k: int = TOP_K) -> tuple:
    """
    Returns
    -------
    jacc_mat : [S, S] pairwise Jaccard
    avg_jacc : scalar
    """
    S = importance_matrix.shape[0]
    top_sets = [set(np.argsort(importance_matrix[i])[-k:]) for i in range(S)]
    jacc_mat = np.eye(S)
    scores   = []
    for i, j in combinations(range(S), 2):
        inter = len(top_sets[i] & top_sets[j])
        union = len(top_sets[i] | top_sets[j])
        j_ij  = inter / union if union > 0 else 0.0
        jacc_mat[i, j] = jacc_mat[j, i] = j_ij
        scores.append(j_ij)
    return jacc_mat, float(np.mean(scores))


def compute_stability_metrics(importance_matrix: np.ndarray) -> dict:
    mean_imp  = importance_matrix.mean(axis=0)     # [C]
    var_imp   = importance_matrix.var(axis=0)      # [C]
    spear_mat = spearman_matrix(importance_matrix)
    jacc_mat, avg_jacc = jaccard_topk(importance_matrix)

    mean_corr = spear_mat[np.triu_indices_from(spear_mat, k=1)].mean()

    return {
        "mean_importance": mean_imp,
        "variance":        var_imp,
        "spearman_matrix": spear_mat,
        "jaccard_matrix":  jacc_mat,
        "mean_spearman":   float(mean_corr),
        "avg_jaccard":     avg_jacc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 – EMOTION-SPECIFIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_emotion_matrices(attr_per_subject: dict) -> dict:
    """
    attr_per_subject : { subject_id : {"attrs": [...], "labels": [...]} }
    returns          : { emotion_name: importance_matrix [S, C] }
    """
    subject_ids = sorted(attr_per_subject.keys())
    emotion_matrices = {}

    for emo_id, emo_name in EMOTION_MAP.items():
        rows = []
        for sid in subject_ids:
            d       = attr_per_subject[sid]
            indices = [i for i, lbl in enumerate(d["labels"]) if lbl == emo_id]
            if not indices:
                rows.append(np.zeros(N_CHANNELS))
                continue
            emo_attrs = [d["attrs"][i] for i in indices]
            rows.append(attribution_to_channel_importance(emo_attrs))
        emotion_matrices[emo_name] = np.stack(rows, axis=0)   # [S, C]

    return emotion_matrices


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 – VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bar_importance(mean_imp, title, fname):
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(range(N_CHANNELS), mean_imp, color="steelblue")
    ax.set_xticks(range(N_CHANNELS))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean |IG| Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_bar_variance(var_imp, title, fname):
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(range(N_CHANNELS), var_imp, color="tomato")
    ax.set_xticks(range(N_CHANNELS))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Variance across Subjects")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_spearman_heatmap(corr_mat, subject_ids, fname):
    labels = [f"S{s}" for s in subject_ids]
    fig, ax = plt.subplots(figsize=(max(8, len(subject_ids)), max(6, len(subject_ids)-2)))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, cmap="RdYlGn", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pairwise Spearman Correlation of Channel Importance Rankings")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_topomap_mne(values, title, fname, kind="mean"):
    """
    Renders a SEED-IV 62-channel topomap using MNE.
    Falls back to a bar plot if MNE / montage issues arise.
    """
    try:
        import mne
        # Build a standard 10-20 montage for the SEED-IV channels
        montage  = mne.channels.make_standard_montage("standard_1020")
        
        # MNE is case-sensitive. We need to find the correct casing from montage
        montage_names_upper = {ch.upper(): ch for ch in montage.ch_names}
        
        valid_ch_names = []
        valid_vals = []
        for i, ch in enumerate(CHANNEL_NAMES):
            ch_upper = ch.upper()
            if ch_upper in montage_names_upper:
                valid_ch_names.append(montage_names_upper[ch_upper])
                valid_vals.append(values[i])

        vals = np.array(valid_vals)
        info = mne.create_info(ch_names=valid_ch_names, sfreq=200, ch_types="eeg")
        info.set_montage(montage)

        evoked = mne.EvokedArray(vals[:, np.newaxis], info)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im, _   = mne.viz.plot_topomap(vals, evoked.info, axes=ax,
                                        show=False, cmap="RdYlGn",
                                        vlim=(vals.min(), vals.max()))
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved topomap: {fname}")

    except Exception as e:
        print(f"  [WARN] MNE topomap failed ({e}) – saving bar plot instead.")
        label = "Mean Importance" if kind == "mean" else "Variance"
        plot_bar_importance(values, title, fname) if kind == "mean" \
            else plot_bar_variance(values, title, fname)


def generate_all_plots(metrics: dict, subject_ids: list,
                       emotion_matrices: dict):
    print("\n── Generating plots ──")

    # 1. Mean importance bar
    plot_bar_importance(
        metrics["mean_importance"],
        "Mean Channel Importance (all subjects)",
        PLOTS_DIR / "mean_importance_bar.png",
    )

    # 2. Variance bar
    plot_bar_variance(
        metrics["variance"],
        "Channel Importance Variance across Subjects",
        PLOTS_DIR / "variance_bar.png",
    )

    # 3. Spearman heatmap
    plot_spearman_heatmap(
        metrics["spearman_matrix"],
        subject_ids,
        PLOTS_DIR / "spearman_heatmap.png",
    )

    # 4. Topographic maps (MNE)
    plot_topomap_mne(
        metrics["mean_importance"],
        "Topomap – Mean Channel Importance",
        PLOTS_DIR / "topomap_mean.png",
        kind="mean",
    )
    plot_topomap_mne(
        metrics["variance"],
        "Topomap – Channel Importance Variance",
        PLOTS_DIR / "topomap_variance.png",
        kind="var",
    )

    # 5. Per-emotion bar plots
    for emo_name, mat in emotion_matrices.items():
        mean_e = mat.mean(axis=0)
        plot_bar_importance(
            mean_e,
            f"Mean Channel Importance – {emo_name}",
            PLOTS_DIR / f"mean_importance_{emo_name}.png",
        )
        var_e = mat.var(axis=0)
        plot_bar_variance(
            var_e,
            f"Channel Importance Variance – {emo_name}",
            PLOTS_DIR / f"variance_{emo_name}.png",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 6 – SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(metrics: dict, importance_matrix: np.ndarray,
                    subject_ids: list):
    mean_imp = metrics["mean_importance"]
    var_imp  = metrics["variance"]

    # Ranked lists
    stable_idx   = np.argsort(var_imp)[:10]         # lowest variance
    variable_idx = np.argsort(var_imp)[-10:][::-1]  # highest variance

    # CSV
    df_channels = pd.DataFrame({
        "channel":          CHANNEL_NAMES,
        "mean_importance":  mean_imp,
        "variance":         var_imp,
    }).sort_values("variance")
    df_channels.to_csv(RESULTS_DIR / "channel_stability.csv", index=False)

    # Subject-level importance matrix CSV
    df_matrix = pd.DataFrame(importance_matrix,
                              index=[f"S{s}" for s in subject_ids],
                              columns=CHANNEL_NAMES)
    df_matrix.to_csv(RESULTS_DIR / "importance_matrix.csv")

    # Markdown
    md_lines = [
        "# EEG Channel Importance Stability Report",
        "",
        "## Dataset",
        f"- Subjects analysed : {len(subject_ids)}",
        f"- Channels           : {N_CHANNELS}",
        "",
        "## Cross-Subject Stability Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Spearman correlation | {metrics['mean_spearman']:.4f} |",
        f"| Average Jaccard (top-{TOP_K}) | {metrics['avg_jaccard']:.4f} |",
        "",
        "## Top 10 Stable Channels (lowest variance)",
        "| Rank | Channel | Variance | Mean Importance |",
        "|------|---------|----------|-----------------|",
    ]
    for rank, idx in enumerate(stable_idx, 1):
        md_lines.append(
            f"| {rank} | {CHANNEL_NAMES[idx]} | {var_imp[idx]:.6f} | {mean_imp[idx]:.6f} |"
        )

    md_lines += [
        "",
        "## Top 10 Variable Channels (highest variance)",
        "| Rank | Channel | Variance | Mean Importance |",
        "|------|---------|----------|-----------------|",
    ]
    for rank, idx in enumerate(variable_idx, 1):
        md_lines.append(
            f"| {rank} | {CHANNEL_NAMES[idx]} | {var_imp[idx]:.6f} | {mean_imp[idx]:.6f} |"
        )

    md_lines += [
        "",
        "## Interpretation",
        "- **High Spearman correlation** (close to 1.0) indicates consistent channel",
        "  rankings across subjects → generalizable biomarker.",
        "- **High Jaccard score** means the same top channels appear for most subjects.",
        "- Channels with **low variance** are good candidates for subject-independent",
        "  EEG biomarkers of emotion.",
        "",
        "## Files Generated",
        "| File | Description |",
        "|------|-------------|",
        "| `results/channel_stability.csv` | Per-channel mean importance + variance |",
        "| `results/importance_matrix.csv` | Subject × Channel importance matrix |",
        "| `results/importance_matrix.npy` | Same, as NumPy array |",
        "| `plots/mean_importance_bar.png` | Bar chart – mean importance |",
        "| `plots/variance_bar.png`        | Bar chart – variance |",
        "| `plots/spearman_heatmap.png`    | Subject correlation heatmap |",
        "| `plots/topomap_mean.png`        | MNE topomap – mean |",
        "| `plots/topomap_variance.png`    | MNE topomap – variance |",
    ]

    md_path = RESULTS_DIR / "stability_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMarkdown report → {md_path}")
    print(f"CSV summary     → {RESULTS_DIR / 'channel_stability.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load training results from step 2
    results_path = RESULTS_DIR / "training_results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Run step2_train.py first. ({results_path} not found)")
    with open(results_path, "rb") as f:
        training_results = pickle.load(f)
    print(f"Loaded results for {len(training_results)} subjects.")

    # ── PART 1: Compute attributions ─────────────────────────────────────────
    attr_cache_path = RESULTS_DIR / "attr_cache.pkl"
    if attr_cache_path.exists():
        print("\n── PART 1: Loading cached Integrated Gradients ──")
        with open(attr_cache_path, "rb") as f:
            attr_per_subject = pickle.load(f)
    else:
        print("\n── PART 1: Computing Integrated Gradients ──")
        attr_per_subject = {}

        for res in training_results:
            sid    = res["subject_id"]
            cfg    = res["model_config"]
            print(f"  Subject {sid} …", end=" ", flush=True)

            model = ShallowConvNet(n_channels=cfg["n_channels"],
                           n_times=cfg["n_times"],
                           n_classes=cfg["n_classes"]).to(device)
            model.load_state_dict(
                torch.load(res["model_path"], map_location=device)
            )
            model.eval()

            normalized_loader = normalize_test_loader(res["test_loader"])
            attrs, preds, labels = compute_subject_attributions(
                model, normalized_loader, device=device
            )
            attr_per_subject[sid] = {
                "attrs":  attrs,
                "preds":  preds,
                "labels": labels,
            }
            print(f"{len(attrs)} samples processed.")
            
        print("\n  Saving computed attributions to cache…")
        with open(attr_cache_path, "wb") as f:
            pickle.dump(attr_per_subject, f)

    # ── PART 2: Aggregation ───────────────────────────────────────────────────
    print("\n── PART 2: Aggregating to subject-level importance ──")
    subject_ids = sorted(attr_per_subject.keys())
    importance_rows = []
    for sid in subject_ids:
        imp = attribution_to_channel_importance(attr_per_subject[sid]["attrs"])
        importance_rows.append(imp)
    importance_matrix = np.stack(importance_rows, axis=0)   # [S, C]
    print(f"  Importance matrix shape: {importance_matrix.shape}")

    np.save(RESULTS_DIR / "importance_matrix.npy", importance_matrix)

    # ── PART 3: Stability metrics ─────────────────────────────────────────────
    print("\n── PART 3: Computing stability metrics ──")
    metrics = compute_stability_metrics(importance_matrix)
    print(f"  Mean Spearman correlation : {metrics['mean_spearman']:.4f}")
    print(f"  Average Jaccard (top-{TOP_K}) : {metrics['avg_jaccard']:.4f}")

    # ── PART 4: Emotion-specific analysis ────────────────────────────────────
    print("\n── PART 4: Per-emotion analysis ──")
    emotion_matrices = compute_emotion_matrices(attr_per_subject)
    for emo, mat in emotion_matrices.items():
        mean_e = mat.mean(axis=0)
        var_e  = mat.var(axis=0)
        _, avg_j = jaccard_topk(mat)
        corr_e   = spearman_matrix(mat)
        mean_c   = corr_e[np.triu_indices_from(corr_e, k=1)].mean()
        print(f"  {emo:8s} | mean_spearman={mean_c:.4f}  avg_jaccard={avg_j:.4f}  "
              f"top_channel={CHANNEL_NAMES[mean_e.argmax()]}")

    # ── PART 5: Plots ─────────────────────────────────────────────────────────
    print("\n── PART 5: Generating visualisations ──")
    generate_all_plots(metrics, subject_ids, emotion_matrices)

    # ── PART 6: Report ────────────────────────────────────────────────────────
    print("\n── PART 6: Generating summary report ──")
    generate_report(metrics, importance_matrix, subject_ids)

    # Save all metrics
    with open(RESULTS_DIR / "stability_metrics.pkl", "wb") as f:
        pickle.dump({
            "metrics":           metrics,
            "importance_matrix": importance_matrix,
            "subject_ids":       subject_ids,
            "emotion_matrices":  emotion_matrices,
        }, f)
    print("\nAll results saved.")
    print(f"  plots/        → {PLOTS_DIR.resolve()}")
    print(f"  results/      → {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()