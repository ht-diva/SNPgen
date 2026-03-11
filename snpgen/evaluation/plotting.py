"""
Visualization functions for evaluation results.
"""

from typing import Dict, List, Optional, Tuple, Union

import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

try:
    from IPython.display import display
    DISPLAY_AVAIL = True
except ImportError:
    DISPLAY_AVAIL = False


# =============================================================================
# Default metric configurations for classification and multiclass tasks
# =============================================================================

# Classification metrics (binary outcomes)
CLASSIFICATION_METRICS = [
    'balanced_accuracy',
    'roc_auc',
]

CLASSIFICATION_METRIC_PRETTY_NAMES = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'f1_score': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall',
    'roc_auc': 'AUC',
    'precision_recall_auc': 'PR-AUC',
    'true_positive_count': 'True Positives',
    'false_positive_count': 'False Positives',
    'true_negative_count': 'True Negatives',
    'false_negative_count': 'False Negatives',
    'mathews_correlation_coefficient': 'MCC',
    'cohens_kappa': "Cohen's Kappa",
}

# Multiclass classification metrics
MULTICLASS_METRICS = [
    'balanced_accuracy',
    'macro_auc',
    'log_loss',
]

MULTICLASS_METRIC_PRETTY_NAMES = {
    'log_loss': 'Log-Loss',
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'macro_auc': 'Macro AUC',
    'weighted_auc': 'Weighted AUC',
    'mcfadden_pseudo_r2': "McFadden's Pseudo-R²",
    'f1_macro': 'Macro F1',
    'f1_weighted': 'Weighted F1',
    'cohens_kappa': "Cohen's Kappa",
    'ordinal_mae': 'Ordinal MAE',
}

# Combined pretty names for convenience
ALL_METRIC_PRETTY_NAMES = {
    **CLASSIFICATION_METRIC_PRETTY_NAMES,
    **MULTICLASS_METRIC_PRETTY_NAMES,
}


def detect_task_type_from_df(df: pd.DataFrame) -> str:
    """Detect task type (classification or multiclass) from a results DataFrame.

    Inspects the columns of the DataFrame to determine whether the results
    contain classification metrics (e.g., roc_auc) or multiclass metrics (e.g., macro_auc).

    Args:
        df: Results DataFrame containing metric columns.

    Returns:
        'classification' or 'multiclass'.

    Raises:
        ValueError: If task type cannot be determined from available metrics.
    """
    columns = set(df.columns)

    # Check for multiclass-specific metrics (must check before classification)
    multiclass_indicators = {'log_loss', 'macro_auc', 'weighted_auc', 'mcfadden_pseudo_r2',
                             'f1_macro', 'f1_weighted', 'ordinal_mae'}

    # Check for classification-specific metrics (binary)
    classification_indicators = {'roc_auc', 'precision_recall_auc'}

    has_multiclass = bool(columns & multiclass_indicators)
    has_classification = bool(columns & classification_indicators)

    # Priority: multiclass > classification
    if has_multiclass:
        return 'multiclass'
    elif has_classification:
        return 'classification'
    else:
        raise ValueError(
            f"Cannot determine task type from columns: {columns}. "
            "Expected classification metrics (roc_auc) or multiclass metrics "
            "(log_loss, macro_auc)."
        )


def get_default_metrics(task_type: str) -> List[str]:
    """Get default metrics to plot for a given task type.

    Args:
        task_type: 'classification' or 'multiclass'.

    Returns:
        List of default metric names.
    """
    if task_type == 'classification':
        return CLASSIFICATION_METRICS.copy()
    elif task_type == 'multiclass':
        return MULTICLASS_METRICS.copy()
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def set_bold(txt: str) -> str:
    """Format text as bold for matplotlib.

    Args:
        txt: Text to format.

    Returns:
        Bold-formatted text for matplotlib.
    """
    txt = str(txt).replace(" ", " \\ ")
    return r"$\bf{" + txt + "}$"


def add_value_labels(
    ax: matplotlib.axes.Axes,
    spacing: int = 5,
    precision: int = 3,
    append_label: str = '',
    errors_df: pd.DataFrame = None,
    fontsize: int = 15,
    color: str = '#b8afae',
) -> None:
    """Add labels to the end of each bar in a bar chart.

    Args:
        ax: The matplotlib axes containing the bar chart.
        spacing: Distance in points between bar and label.
        precision: Number of decimal places for labels.
        append_label: String to append to each label.
        errors_df: DataFrame with error values (CI or std) to account
            for label position.
        fontsize: Font size for labels.
        color: Color for labels.
    """
    for i, rect in enumerate(ax.patches):
        y_value = rect.get_height()
        y_pos = y_value
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = 'bottom'

        if errors_df is not None:
            row_idx = i % len(errors_df)
            col_idx = i // len(errors_df)
            try:
                err_value = errors_df.iloc[row_idx, col_idx]
                y_pos += err_value
            except:
                pass

        if y_pos < 0:
            space *= -1
            va = 'top'

        if y_value.is_integer():
            label = f"{y_value:.0f}{append_label}"
        else:
            label = f"{y_value:.{precision}f}{append_label}"

        ax.annotate(
            label,
            (x_value, y_pos),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            rotation=0,
            color=color,
            fontsize=fontsize,
            va=va
        )


def calculate_ci(group: pd.Series, confidence: float = 0.95) -> float:
    """Calculate confidence interval margin of error for a group.

    Args:
        group: Series of values.
        confidence: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Margin of error for the confidence interval.
    """
    n = len(group)
    mean = group.mean()
    std = group.std(ddof=1)

    # Get t-value for given confidence level and degrees of freedom
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    # Calculate margin of error
    margin_of_error = t_value * (std / np.sqrt(n))

    return margin_of_error


def plot_metrics_with_ci(
    result_df: pd.DataFrame,
    metric_pretty_names: Optional[Dict[str, str]] = None,
    model_pretty_names: Optional[Dict[str, str]] = None,
    colors: Optional[List[str]] = None,
    confidence: float = 0.95,
    joint: bool = True,
    orientation: str = 'horizontal',
    base_height: int = 6,
    base_width: int = 12,
    show_prs_baseline: bool = True,
    show_gwas_prs_baseline: bool = True,
    gwas_prs_color: str = '#E63946',
    gwas_prs_alpha: float = 0.7,
    gwas_prs_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = None,
    metrics_to_plot: Optional[List[str]] = None,
    task_type: str = 'auto',
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[matplotlib.figure.Figure]:
    """Plot metrics comparison with confidence intervals.

    Supports automatic detection of task type (classification vs multiclass)
    and uses appropriate default metrics when not specified.

    Args:
        result_df: DataFrame with MultiIndex (name, fold, model) containing metrics.
        metric_pretty_names: Dictionary mapping metric names to display names.
            If None, uses built-in defaults based on detected task type.
        model_pretty_names: Dictionary mapping model names to display names.
            If None, uses the raw model names.
        colors: List of colors for each split (e.g., Real, Synthetic).
            If None, uses matplotlib default color cycle.
        confidence: Confidence level for CI. Set to None to use std instead.
        joint: If True, plot all metrics in one figure. If False, separate figures.
        orientation: 'horizontal' or 'vertical' layout for joint plots.
        base_height: Base height for each subplot.
        base_width: Base width for each subplot.
        show_prs_baseline: Whether to show PRS Univariate baseline as horizontal lines.
        show_gwas_prs_baseline: Whether to show GWAS PRS baseline as horizontal lines.
        gwas_prs_color: Color for GWAS PRS reference line.
        gwas_prs_alpha: Alpha (opacity) for GWAS PRS reference line.
        gwas_prs_df: Optional DataFrame with GWAS PRS results for reference line.
            If None and show_gwas_prs_baseline=True, looks for GWAS model in result_df.
        figsize: Optional explicit figure size (width, height).
        metrics_to_plot: List of metric columns to plot. If None, auto-selects
            based on task type (e.g., ['balanced_accuracy', 'roc_auc'] for
            classification, ['balanced_accuracy', 'macro_auc'] for multiclass).
        task_type: 'auto' to detect from DataFrame, or explicitly set
            'classification' or 'multiclass'.
        title: Optional title for the figure. If None, no title is displayed.
        output_path: Optional file path where the generated plot is saved.
            When provided, a CSV with plotting data is also exported using the
            same base filename and `.csv` extension.

    Returns:
        Figure object if joint=True, None otherwise.
    """
    csv_rows: List[Dict[str, Union[str, float, int, None]]] = []
    csv_output_path: Optional[str] = None
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        csv_output_path = f"{os.path.splitext(output_path)[0]}.csv"

    # Auto-detect task type if needed
    if task_type == 'auto':
        detected_task = detect_task_type_from_df(result_df)
    else:
        detected_task = task_type

    # Set default metrics to plot based on task type
    if metrics_to_plot is None:
        default_metrics = get_default_metrics(detected_task)
        # Filter to only metrics that exist in the DataFrame
        metrics_to_plot = [m for m in default_metrics if m in result_df.columns]
        if not metrics_to_plot:
            # Fallback: use first available metrics
            metrics_to_plot = result_df.columns[:2].tolist()
        print(f"Auto-detected task type: {detected_task}")
        print(f"Plotting metrics: {metrics_to_plot}")

    selected_metrics = [m for m in metrics_to_plot if m in result_df.columns]
    if not selected_metrics:
        raise ValueError(
            f"None of the requested metrics were found in result_df columns. "
            f"Requested: {metrics_to_plot}. Available: {list(result_df.columns)}"
        )

    # Filter DataFrame to only include requested metrics
    result_df = result_df[selected_metrics]

    # Set default pretty names
    if metric_pretty_names is None:
        metric_pretty_names = ALL_METRIC_PRETTY_NAMES

    # Set default model pretty names (identity mapping)
    if model_pretty_names is None:
        model_pretty_names = {}

    # Set default colors from matplotlib color cycle
    if colors is None:
        n_splits = result_df.index.get_level_values(0).nunique()
        colors = [plt.cm.tab10(i) for i in range(n_splits)]
    # Calculate mean and std across folds
    # level 0 is split (Real/Syn), level 2 is model (level 1 was fold)
    mean_df = result_df.groupby(level=[0, 2], sort=False).mean()
    std_df = result_df.groupby(level=[0, 2], sort=False).std()
    count_df = result_df.groupby(level=[0, 2], sort=False).count()

    if confidence is not None:
        ci_df = result_df.groupby(level=[0, 2], sort=False).agg(
            lambda x: calculate_ci(x, confidence=confidence)
        )

    if joint:
        n_metrics = len(mean_df.columns)

        if figsize:
            fig, axes = plt.subplots(
                n_metrics if orientation == 'vertical' else (n_metrics + 1) // 2,
                1 if orientation == 'vertical' else 2,
                figsize=figsize
            )
        elif orientation == 'horizontal':
            n_rows = (n_metrics + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(base_width * 2, base_height * n_rows))
        else:
            fig, axes = plt.subplots(n_metrics, 1, figsize=(base_width, base_height * n_metrics))

        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

    legend_labels = None
    legend_handles = None
    error_method = 'std' if confidence is None else f'ci_{int(confidence * 100)}'

    for idx, metric in enumerate(mean_df.columns):
        if not joint:
            print(f"====================== {metric_pretty_names.get(metric, metric)} ======================")

        if joint:
            current_fig = fig
            ax = axes[idx]
        else:
            current_fig, ax = plt.subplots(figsize=(12, 4))

        unstacked_mean_raw = mean_df[metric].unstack(level=0)
        unstacked_std_raw = std_df[metric].unstack(level=0)
        unstacked_count_raw = count_df[metric].unstack(level=0)

        unstacked_mean = unstacked_mean_raw.rename(index=model_pretty_names)
        unstacked_std = unstacked_std_raw.rename(index=model_pretty_names)

        if confidence is None:
            errors_df_raw = unstacked_std_raw
            errors_df = unstacked_std
        else:
            unstacked_ci_raw = ci_df[metric].unstack(level=0)
            errors_df_raw = unstacked_ci_raw
            unstacked_ci = unstacked_ci_raw.rename(index=model_pretty_names)
            errors_df = unstacked_ci

        for raw_model in unstacked_mean_raw.index:
            display_model = model_pretty_names.get(raw_model, raw_model)
            for split_name in unstacked_mean_raw.columns:
                mean_val = unstacked_mean_raw.loc[raw_model, split_name]
                err_val = errors_df_raw.loc[raw_model, split_name]
                std_val = unstacked_std_raw.loc[raw_model, split_name]
                n_val = unstacked_count_raw.loc[raw_model, split_name]
                csv_rows.append(
                    {
                        'row_kind': 'bar',
                        'reference_name': '',
                        'metric': metric,
                        'split': split_name,
                        'model_raw': raw_model,
                        'model_display': display_model,
                        'value': mean_val,
                        'error': err_val,
                        'value_lower': mean_val - err_val,
                        'value_upper': mean_val + err_val,
                        'std': std_val,
                        'n': int(n_val) if pd.notna(n_val) else None,
                        'error_method': error_method,
                        'confidence': confidence,
                    }
                )

        bars = unstacked_mean.plot(
            kind='bar', ax=ax, color=colors, legend=False, width=0.65,
            yerr=errors_df, capsize=4
        )

        handles, labels = ax.get_legend_handles_labels()

        if not joint and DISPLAY_AVAIL:
            display(unstacked_mean)

        # Add PRS baseline if requested (no legend entry)
        if show_prs_baseline:
            prs_key = next((key for key in unstacked_mean.index if 'PRS' in key and 'gwas' not in key.lower()), None)
            if prs_key:
                prs_mean = unstacked_mean.loc[prs_key].iloc[0]
                prs_ci = errors_df.loc[prs_key].iloc[0]

                prs_raw_key = next(
                    (
                        raw_key for raw_key in unstacked_mean_raw.index
                        if model_pretty_names.get(raw_key, raw_key) == prs_key
                    ),
                    prs_key if prs_key in unstacked_mean_raw.index else None
                )
                prs_n = None
                if prs_raw_key is not None and prs_raw_key in unstacked_count_raw.index:
                    prs_n = int(unstacked_count_raw.loc[prs_raw_key].iloc[0])

                csv_rows.append(
                    {
                        'row_kind': 'reference',
                        'reference_name': 'prs_baseline',
                        'metric': metric,
                        'split': '',
                        'model_raw': prs_raw_key if prs_raw_key is not None else '',
                        'model_display': prs_key,
                        'value': prs_mean,
                        'error': prs_ci,
                        'value_lower': prs_mean - prs_ci,
                        'value_upper': prs_mean + prs_ci,
                        'std': np.nan,
                        'n': prs_n,
                        'error_method': error_method,
                        'confidence': confidence,
                    }
                )

                ax.axhline(y=prs_mean, color='black', linestyle='-', alpha=0.2, linewidth=1)
                ax.axhline(y=prs_mean - prs_ci, color='blue', linestyle='--', alpha=0.3, linewidth=1)
                ax.axhline(y=prs_mean + prs_ci, color='blue', linestyle='--', alpha=0.3, linewidth=1)

        # Add GWAS PRS baseline if requested (as red reference line)
        if show_gwas_prs_baseline:
            gwas_mean = None
            gwas_ci = None
            gwas_prs_key = None

            # First try to use external gwas_prs_df if provided
            if gwas_prs_df is not None:
                # Check if gwas_prs_df contains multiple models (has 'model' in index)
                if isinstance(gwas_prs_df.index, pd.MultiIndex) and 'model' in gwas_prs_df.index.names:
                    # Multiple models - select the preferred one
                    gwas_models = [m for m in gwas_prs_df.index.get_level_values('model').unique()
                                   if 'gwas' in m.lower()]
                    if gwas_models:
                        print(f"GWAS PRS models available: {gwas_models}")
                        gwas_prs_key = None
                        for preferred in ['scaled', 'overall']:
                            gwas_prs_key = next((k for k in gwas_models if preferred in k.lower()), None)
                            if gwas_prs_key:
                                break
                        if gwas_prs_key is None:
                            print(f"No preferred GWAS PRS variant found; using first available.")
                            gwas_prs_key = gwas_models[0]
                        print(f"Using GWAS PRS baseline from model: {gwas_prs_key}")
                        # Extract the selected model's data
                        selected_df = gwas_prs_df.xs(gwas_prs_key, level='model')
                        if metric in selected_df.columns:
                            gwas_mean = selected_df[metric].mean()
                            if confidence is not None:
                                gwas_ci = calculate_ci(selected_df[metric], confidence=confidence)
                            else:
                                gwas_ci = selected_df[metric].std()
                elif metric in gwas_prs_df.columns:
                    # Single model's data - use directly
                    gwas_mean = gwas_prs_df[metric].mean()
                    if confidence is not None:
                        gwas_ci = calculate_ci(gwas_prs_df[metric], confidence=confidence)
                    else:
                        gwas_ci = gwas_prs_df[metric].std()

            # Fall back to looking in the result_df if not found
            if gwas_mean is None:
                # Prefer 'scaled' for classification, 'overall' for multiclass
                gwas_models = [key for key in unstacked_mean.index if 'gwas' in key.lower()]
                gwas_prs_key = None
                if gwas_models:
                    print(f"GWAS PRS models available: {gwas_models}")
                    # Try to find the preferred variant
                    for preferred in ['scaled', 'overall']:
                        gwas_prs_key = next((k for k in gwas_models if preferred in k.lower()), None)
                        if gwas_prs_key:
                            break
                    # Fall back to first available if no preferred found
                    if gwas_prs_key is None:
                        print(f"No preferred GWAS PRS variant found; using first available.")
                        gwas_prs_key = gwas_models[0]
                if gwas_prs_key:
                    print(f"Using GWAS PRS baseline from model: {gwas_prs_key}")
                    gwas_mean = unstacked_mean.loc[gwas_prs_key].iloc[0]
                    gwas_ci = errors_df.loc[gwas_prs_key].iloc[0]

            if gwas_mean is not None:
                csv_rows.append(
                    {
                        'row_kind': 'reference',
                        'reference_name': 'gwas_prs_baseline',
                        'metric': metric,
                        'split': '',
                        'model_raw': gwas_prs_key if gwas_prs_key is not None else '',
                        'model_display': gwas_prs_key if gwas_prs_key is not None else '',
                        'value': gwas_mean,
                        'error': gwas_ci,
                        'value_lower': gwas_mean - gwas_ci,
                        'value_upper': gwas_mean + gwas_ci,
                        'std': np.nan,
                        'n': None,
                        'error_method': error_method,
                        'confidence': confidence,
                    }
                )

                # Main reference line (solid, semi-transparent)
                ax.axhline(y=gwas_mean, color=gwas_prs_color, linestyle='-',
                          alpha=gwas_prs_alpha, linewidth=1, label='PRS GWAS')
                # CI lines (dashed, more transparent)
                ax.axhline(y=gwas_mean - gwas_ci, color=gwas_prs_color,
                          linestyle='--', alpha=gwas_prs_alpha * 0.5, linewidth=1)
                ax.axhline(y=gwas_mean + gwas_ci, color=gwas_prs_color,
                          linestyle='--', alpha=gwas_prs_alpha * 0.5, linewidth=1)

        ax.set_ylabel(
            metric_pretty_names.get(metric, metric),
            fontsize=16, labelpad=15, fontdict=dict(weight='bold')
        )
        ax.set_xlabel("")
        ax.set_ylim(ymin=unstacked_mean.min().min() - unstacked_mean.min().min() * 0.1)
        ax.tick_params(axis='x', rotation=25, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        precision = 0 if pd.api.types.is_integer_dtype(result_df[metric].dtype) else 3
        add_value_labels(ax, precision=precision, spacing=10, errors_df=errors_df)

        # Capture legend handles/labels (including reference lines)
        if legend_labels is None:
            # Get all handles and labels (bars + reference lines)
            all_handles, all_labels = ax.get_legend_handles_labels()
            # Separate bar handles (colors) from reference line handles
            legend_handles = all_handles
            legend_labels = all_labels

        if not joint:
            ax.legend(
                legend_handles, legend_labels,
                loc='upper center', bbox_to_anchor=(0.5, -0.3),
                fancybox=True, shadow=False, ncol=5, fontsize=12
            )
            plt.tight_layout()
            if output_path is not None:
                current_fig.savefig(output_path, bbox_inches='tight')
            plt.show()

    if joint:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.51, -0.03),
            ncol=5, bbox_transform=fig.transFigure, fontsize=12
        )

        if orientation == 'horizontal' and len(result_df.columns) % 2 != 0:
            axes[-1].set_visible(False)

        # Add title if provided
        if title is not None:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.)

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, bbox_inches='tight')
        if csv_output_path is not None:
            csv_columns = [
                'row_kind', 'reference_name', 'metric', 'split', 'model_raw',
                'model_display', 'value', 'error', 'value_lower', 'value_upper',
                'std', 'n', 'error_method', 'confidence'
            ]
            pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_output_path, index=False)
        plt.show()
        return fig

    if csv_output_path is not None:
        csv_columns = [
            'row_kind', 'reference_name', 'metric', 'split', 'model_raw',
            'model_display', 'value', 'error', 'value_lower', 'value_upper',
            'std', 'n', 'error_method', 'confidence'
        ]
        pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_output_path, index=False)

    return None


# =============================================================================
# Multiclass-specific plotting functions
# =============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot a confusion matrix heatmap for multiclass classification.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        class_names: Optional list of class names for axis labels.
        normalize: If True, normalize by true labels (rows sum to 1).
        figsize: Figure size if creating new figure.
        cmap: Colormap name.
        title: Plot title.
        ax: Optional existing axes to plot on.

    Returns:
        Matplotlib axes with the confusion matrix plot.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


def plot_class_proportions_by_decile(
    prs: np.ndarray,
    y_true: np.ndarray,
    n_deciles: int = 10,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis',
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot class proportions as stacked bars by PRS decile.

    Useful for visualizing how class distribution changes across PRS quantiles.

    Args:
        prs: Polygenic risk scores (1D array for single PRS, or argmax of 2D).
        y_true: True class labels.
        n_deciles: Number of deciles/quantiles to bin PRS into.
        class_names: Optional list of class names for legend.
        figsize: Figure size if creating new figure.
        cmap: Colormap name for class colors.
        ax: Optional existing axes to plot on.

    Returns:
        Matplotlib axes with the stacked bar plot.
    """
    # If PRS is 2D (multiclass probabilities), take predicted class or use first column
    if prs.ndim == 2:
        prs_1d = prs.argmax(axis=1)
    else:
        prs_1d = prs

    n_classes = len(np.unique(y_true))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Bin PRS into deciles
    decile_labels = pd.qcut(prs_1d, q=n_deciles, labels=False, duplicates='drop')
    actual_deciles = len(np.unique(decile_labels))

    # Calculate class proportions per decile
    proportions = np.zeros((actual_deciles, n_classes))
    for d in range(actual_deciles):
        mask = decile_labels == d
        if mask.sum() > 0:
            for c in range(n_classes):
                proportions[d, c] = np.sum(y_true[mask] == c) / mask.sum()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar chart
    colors = plt.cm.get_cmap(cmap, n_classes)
    bottom = np.zeros(actual_deciles)
    decile_positions = np.arange(actual_deciles)

    for c in range(n_classes):
        ax.bar(decile_positions, proportions[:, c], bottom=bottom,
               label=class_names[c], color=colors(c), width=0.8)
        bottom += proportions[:, c]

    ax.set_xlabel('PRS Decile', fontsize=12)
    ax.set_ylabel('Class Proportion', fontsize=12)
    ax.set_title('Class Proportions by PRS Decile', fontsize=14, fontweight='bold')
    ax.set_xticks(decile_positions)
    ax.set_xticklabels([f'{i+1}' for i in range(actual_deciles)])
    ax.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return ax


def plot_ordinal_trend_by_decile(
    prs: np.ndarray,
    y_true: np.ndarray,
    n_deciles: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    show_ci: bool = True,
    confidence: float = 0.95,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot mean class (ordinal) by PRS decile with confidence interval.

    Useful for ordinal multiclass where classes have a natural ordering
    (e.g., low/medium/high risk).

    Args:
        prs: Polygenic risk scores (1D or predicted class from 2D).
        y_true: True class labels (ordinal integers 0, 1, 2, ...).
        n_deciles: Number of deciles to bin PRS into.
        figsize: Figure size if creating new figure.
        show_ci: Whether to show confidence interval shading.
        confidence: Confidence level for CI.
        ax: Optional existing axes to plot on.

    Returns:
        Matplotlib axes with the trend line plot.
    """
    # If PRS is 2D, take first column or argmax
    if prs.ndim == 2:
        prs_1d = prs[:, -1]  # Use probability of highest class
    else:
        prs_1d = prs

    # Bin PRS into deciles
    decile_labels = pd.qcut(prs_1d, q=n_deciles, labels=False, duplicates='drop')
    actual_deciles = len(np.unique(decile_labels))

    # Calculate mean and CI per decile
    means = []
    cis = []
    decile_centers = []

    for d in range(actual_deciles):
        mask = decile_labels == d
        if mask.sum() > 0:
            values = y_true[mask]
            mean_val = np.mean(values)
            means.append(mean_val)
            decile_centers.append(d + 1)

            if show_ci and len(values) > 1:
                ci = calculate_ci(pd.Series(values), confidence=confidence)
                cis.append(ci)
            else:
                cis.append(0)

    means = np.array(means)
    cis = np.array(cis)
    decile_centers = np.array(decile_centers)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot line with CI
    ax.plot(decile_centers, means, 'o-', linewidth=2, markersize=8, color='#1f77b4')

    if show_ci:
        ax.fill_between(decile_centers, means - cis, means + cis,
                       alpha=0.2, color='#1f77b4')

    ax.set_xlabel('PRS Decile', fontsize=12)
    ax.set_ylabel('Mean Class (Ordinal)', fontsize=12)
    ax.set_title('Ordinal Trend by PRS Decile', fontsize=14, fontweight='bold')
    ax.set_xticks(decile_centers)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> matplotlib.figure.Figure:
    """Plot one-vs-rest ROC curves for each class.

    Creates a multi-panel figure with one ROC curve per class.

    Args:
        y_true: True class labels (integers 0 to n_classes-1).
        y_prob: Predicted probabilities, shape (n_samples, n_classes).
        class_names: Optional list of class names for titles.
        figsize: Figure size.

    Returns:
        Matplotlib figure with ROC curves.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    n_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Create subplots
    n_cols = min(n_classes, 4)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    if n_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_classes):
        ax = axes[i]

        # One-vs-rest binary labels
        y_binary = (y_true == i).astype(int)

        # Check if we have both classes
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            auc_score = roc_auc_score(y_binary, y_prob[:, i])

            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_names[i]}', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
        else:
            ax.text(0.5, 0.5, 'N/A\n(single class)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{class_names[i]}', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('One-vs-Rest ROC Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
