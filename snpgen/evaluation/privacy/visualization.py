"""
Visualization functions for privacy evaluation results.

Each function takes a result dataclass and produces matplotlib figures.
All plots follow a consistent style with clear labels and interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

DEFAULT_FIGSIZE = (8, 5)
DEFAULT_DPI = 100
COLORS = {
    'syn': '#2196F3',       # Blue
    'holdout': '#FF9800',   # Orange
    'train': '#4CAF50',     # Green
    'recon': '#9C27B0',     # Purple
    'ideal': '#757575',     # Gray
}


def _format_text_table(headers, rows, title=None):
    """Format tabular data as a plain-text aligned table."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(str(h).ljust(widths[idx]) for idx, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    body_lines = [" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in str_rows]

    lines = []
    if title:
        lines.append(str(title))
    lines.append(header_line)
    lines.append(sep_line)
    lines.extend(body_lines)
    return "\n".join(lines)


def _apply_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to an axes."""
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# DCR
# ---------------------------------------------------------------------------

def plot_dcr_distributions(dcr_result, title=None, ax=None, figsize=DEFAULT_FIGSIZE):
    """Plot KDE overlay of synthetic vs holdout DCR distributions.

    Args:
        dcr_result: DCRResult instance.
        title: Optional custom title.
        ax: Matplotlib axes (creates new figure if None).
        figsize: Figure size.

    Returns:
        Figure if ax was None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # KDE plots
    from scipy.stats import gaussian_kde

    dcr_syn = dcr_result.dcr_syn
    dcr_hold = dcr_result.dcr_holdout

    # Compute KDE
    x_min = min(dcr_syn.min(), dcr_hold.min())
    x_max = max(dcr_syn.max(), dcr_hold.max())
    x_range = np.linspace(x_min, x_max, 500)

    kde_syn = gaussian_kde(dcr_syn)
    kde_hold = gaussian_kde(dcr_hold)

    ax.fill_between(x_range, kde_syn(x_range), alpha=0.3, color=COLORS['syn'], label='Synthetic → Train')
    ax.plot(x_range, kde_syn(x_range), color=COLORS['syn'], linewidth=2)

    ax.fill_between(x_range, kde_hold(x_range), alpha=0.3, color=COLORS['holdout'], label='Holdout → Train')
    ax.plot(x_range, kde_hold(x_range), color=COLORS['holdout'], linewidth=2)

    # Mark medians
    ax.axvline(dcr_result.dcr_syn_median, color=COLORS['syn'], linestyle='--', alpha=0.7,
               label=f'Syn median: {dcr_result.dcr_syn_median:.0f}')
    ax.axvline(dcr_result.dcr_holdout_median, color=COLORS['holdout'], linestyle='--', alpha=0.7,
               label=f'Hold median: {dcr_result.dcr_holdout_median:.0f}')

    # Annotation
    ann_text = (f"KS p={dcr_result.ks_pvalue:.2e}\n"
                f"Frac below 5th pct: {dcr_result.frac_below_5th_pct:.4f}")
    ax.text(0.98, 0.95, ann_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    _apply_style(ax,
                 title=title or f'DCR Distribution ({dcr_result.metric})',
                 xlabel=f'Distance to Closest Record ({dcr_result.metric})',
                 ylabel='Density')
    ax.legend(loc='upper left', fontsize=9)

    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NNAA
# ---------------------------------------------------------------------------

def plot_nnaa_summary(nnaa_result, title=None, ax=None, figsize=(6, 4)):
    """Bar chart of NNAA scores.

    Args:
        nnaa_result: NNAAResult instance.
        title: Optional custom title.
        ax: Matplotlib axes.
        figsize: Figure size.

    Returns:
        Figure if ax was None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    values = [nnaa_result.aa_train, nnaa_result.aa_syn, nnaa_result.privacy_score]
    labels = ['AA_train', 'AA_syn', 'Privacy Score']
    colors = [COLORS['train'], COLORS['syn'], COLORS['ideal']]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add ideal line at 0.5
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (0.5)')

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(max(values) + 0.1, 0.7))
    _apply_style(ax,
                 title=title or f'NNAA ({nnaa_result.metric})',
                 ylabel='Accuracy')
    ax.legend(fontsize=9)

    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MI ROC
# ---------------------------------------------------------------------------

def plot_mi_roc(mi_result, title=None, ax=None, figsize=DEFAULT_FIGSIZE):
    """ROC curve and distance distributions for membership inference.

    Args:
        mi_result: MIResult instance.
        title: Optional custom title.
        ax: Matplotlib axes.
        figsize: Figure size.

    Returns:
        Figure if ax was None.
    """
    from sklearn.metrics import roc_curve

    fig = None
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]), dpi=DEFAULT_DPI)
    else:
        axes = [ax, None]

    # Left: distance distributions
    ax1 = axes[0]
    from scipy.stats import gaussian_kde

    d_train = mi_result.dcr_train_to_syn
    d_holdout = mi_result.dcr_holdout_to_syn

    x_min = min(d_train.min(), d_holdout.min())
    x_max = max(d_train.max(), d_holdout.max())
    x_range = np.linspace(x_min, x_max, 500)

    kde_train = gaussian_kde(d_train)
    kde_holdout = gaussian_kde(d_holdout)

    ax1.fill_between(x_range, kde_train(x_range), alpha=0.3, color=COLORS['train'],
                     label=f'Train → Syn (mean={mi_result.dcr_train_mean:.0f})')
    ax1.plot(x_range, kde_train(x_range), color=COLORS['train'], linewidth=2)

    ax1.fill_between(x_range, kde_holdout(x_range), alpha=0.3, color=COLORS['holdout'],
                     label=f'Holdout → Syn (mean={mi_result.dcr_holdout_mean:.0f})')
    ax1.plot(x_range, kde_holdout(x_range), color=COLORS['holdout'], linewidth=2)

    _apply_style(ax1,
                 title='MI: Distance Distributions',
                 xlabel=f'Min Distance ({mi_result.metric})',
                 ylabel='Density')
    ax1.legend(fontsize=9)

    # Right: ROC curve
    if axes[1] is not None:
        ax2 = axes[1]
        labels = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_holdout))])
        scores = np.concatenate([-d_train, -d_holdout])

        fpr, tpr, _ = roc_curve(labels, scores)
        ax2.plot(fpr, tpr, color=COLORS['syn'], linewidth=2,
                 label=f'MI AUC = {mi_result.auc:.4f}')
        ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random (0.5)')

        _apply_style(ax2,
                     title=title or 'MI: ROC Curve',
                     xlabel='False Positive Rate',
                     ylabel='True Positive Rate')
        ax2.legend(fontsize=9, loc='lower right')

    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NNDR
# ---------------------------------------------------------------------------

def plot_nndr_histogram(nndr_result, title=None, ax=None, figsize=DEFAULT_FIGSIZE):
    """Histogram of NNDR values.

    Args:
        nndr_result: NNDRResult instance.
        title: Optional custom title.
        ax: Matplotlib axes.
        figsize: Figure size.

    Returns:
        Figure if ax was None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    nndr = nndr_result.nndr_values

    ax.hist(nndr, bins=100, density=True, alpha=0.7, color=COLORS['syn'],
            edgecolor='black', linewidth=0.3)

    # Mark thresholds
    ax.axvline(0.8, color='orange', linestyle='--', linewidth=1.5,
               label=f'NNDR=0.8 (frac<: {nndr_result.frac_below_08:.4f})')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5,
               label=f'NNDR=0.5 (frac<: {nndr_result.frac_below_05:.4f})')

    # Mark mean/median
    ax.axvline(nndr_result.nndr_median, color='black', linestyle='-', linewidth=1.5,
               label=f'Median: {nndr_result.nndr_median:.4f}')

    _apply_style(ax,
                 title=title or f'NNDR Distribution ({nndr_result.metric})',
                 xlabel='NNDR (d₁ / d₂)',
                 ylabel='Density')
    ax.legend(fontsize=9)

    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MAF Scatter
# ---------------------------------------------------------------------------

def plot_maf_scatter(maf_result, title=None, ax=None, figsize=(6, 6)):
    """Scatter plot of real vs synthetic allele frequencies.

    Args:
        maf_result: MAFResult instance.
        title: Optional custom title.
        ax: Matplotlib axes.
        figsize: Figure size.

    Returns:
        Figure if ax was None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    ax.scatter(maf_result.maf_real, maf_result.maf_syn,
               alpha=0.3, s=8, color=COLORS['syn'])

    # Diagonal
    lims = [0, max(maf_result.maf_real.max(), maf_result.maf_syn.max()) * 1.05]
    ax.plot(lims, lims, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect match')

    # Annotation
    ann = (f"Pearson r = {maf_result.pearson_r:.6f}\n"
           f"Mean |drift| = {maf_result.mean_abs_drift:.6f}\n"
           f"Max |drift| = {maf_result.max_abs_drift:.6f}")
    ax.text(0.05, 0.95, ann, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    _apply_style(ax,
                 title=title or 'Allele Frequency Comparison',
                 xlabel='Real Allele Frequency',
                 ylabel='Synthetic Allele Frequency')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9)

    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def plot_privacy_summary_table(all_results, figsize=(14, None), print_text=True):
    """Create a summary table of all privacy metrics across traits.

    Args:
        all_results: Dict mapping trait_target (e.g., 'T2D_syn') to result dicts.
        figsize: Figure size (height auto-computed if None).
        print_text: Whether to print a formatted text table to stdout.

    Returns:
        Figure.
    """
    # Collect summary rows
    rows = []
    for trait_name, results in all_results.items():
        row = {'Trait': trait_name}

        imr = results.get('imr__overall')
        if imr is not None:
            row['IMR'] = f"{imr.match_rate:.6f}"

        dcr = results.get('dcr__overall')
        if dcr is not None:
            row['DCR Syn Med'] = f"{dcr.dcr_syn_median:.0f}"
            row['DCR Hold Med'] = f"{dcr.dcr_holdout_median:.0f}"
            row['DCR Frac<5pct'] = f"{dcr.frac_below_5th_pct:.4f}"

        nnaa_r = results.get('nnaa__overall')
        if nnaa_r is not None:
            row['NNAA Score'] = f"{nnaa_r.privacy_score:.4f}"

        mi = results.get('mi__overall')
        if mi is not None:
            row['MI AUC'] = f"{mi.auc:.4f}"

        nndr_r = results.get('nndr__overall')
        if nndr_r is not None:
            row['NNDR Med'] = f"{nndr_r.nndr_median:.4f}"

        maf = results.get('maf__overall')
        if maf is not None:
            row['MAF r'] = f"{maf.pearson_r:.6f}"

        rows.append(row)

    if not rows:
        print("No results to display.")
        return None

    # Determine columns
    all_cols = []
    for r in rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    # Build table data
    cell_text = []
    for r in rows:
        cell_text.append([r.get(c, '-') for c in all_cols])

    if print_text:
        print(_format_text_table(all_cols, cell_text, title='Privacy Evaluation Summary'))

    n_rows = len(rows)
    height = figsize[1] if figsize[1] else max(2, 0.5 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(figsize[0], height), dpi=DEFAULT_DPI)
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=all_cols,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold')

    ax.set_title('Privacy Evaluation Summary', fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    return fig
