import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


METRIC_LABELS = {
    "bbox_density": "Bounding-box density",
    "D_est": "Fractal dimension",
    "R_g": "Radius of gyration",
    "height": "Cluster height",
    "max_width": "Maximum width",
    "aspect_ratio": "Aspect ratio",
    "perimeter_per_occupied": "Perimeter / occupied site",
    "perimeter": "Perimeter length",
    "occupancy": "Occupancy fraction",
}

PARAM_LABELS = {
    "ita": r"$\eta$",
    "ps": r"$p_s$",
}


def _get_metric_bounds(metric, df):
    if metric in {"bbox_density", "occupancy"}:
        return 0.0, 1.0
    if metric in {"height", "max_width"}:
        if "N" in df.columns:
            return 0.0, float(df["N"].iloc[0])
        return 0.0, None
    if metric in {"aspect_ratio", "R_g", "D_est", "perimeter", "perimeter_per_occupied"}:
        return 0.0, None
    return None, None


def _clip_band(mean, std, lower=None, upper=None):
    y_low = mean - std
    y_high = mean + std
    if lower is not None:
        y_low = np.maximum(y_low, lower)
        y_high = np.maximum(y_high, lower)
    if upper is not None:
        y_low = np.minimum(y_low, upper)
        y_high = np.minimum(y_high, upper)
    return y_low, y_high


def _prepare_grouped(df, param_col, metric):
    grouped = (
        df.groupby(param_col)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(param_col)
    )
    x = grouped[param_col].to_numpy(dtype=float)
    mean = grouped["mean"].to_numpy(dtype=float)
    std = np.nan_to_num(grouped["std"].to_numpy(dtype=float), nan=0.0)
    return x, mean, std


def plot_metric_comparison(
    metric,
    df_pde,
    df_mc,
    out_path=None,
    pde_param_col="ita",
    mc_param_col="ps",
    metric_label=None,
    pde_panel_title="PDE DLA",
    mc_panel_title="Monte Carlo DLA",
    show_title=False,
    figure_title=None,
    panel_titles=True,
    figsize=(10, 4.5),
    title_fontsize=18,
    panel_title_fontsize=14,
    label_fontsize=13,
    tick_fontsize=11,
    line_width=2.2,
    marker_size=6,
    alpha_band=0.20,
):
    """
    Two-panel comparison plot with shared y-axis and clean labels.
    """

    sns.set_style("whitegrid")

    df_pde = df_pde.copy()
    df_mc = df_mc.copy()

    # clean parameter columns
    df_pde[pde_param_col] = pd.to_numeric(df_pde[pde_param_col], errors="coerce").round(1)
    df_mc[mc_param_col] = pd.to_numeric(df_mc[mc_param_col], errors="coerce")

    df_pde = df_pde.dropna(subset=[pde_param_col])
    df_mc = df_mc.dropna(subset=[mc_param_col])

    metric_label = metric_label or METRIC_LABELS.get(metric, metric)
    pde_xlabel = PARAM_LABELS.get(pde_param_col, pde_param_col)
    mc_xlabel = PARAM_LABELS.get(mc_param_col, mc_param_col)

    # grouped data
    x_pde, mean_pde, std_pde = _prepare_grouped(df_pde, pde_param_col, metric)
    x_mc, mean_mc, std_mc = _prepare_grouped(df_mc, mc_param_col, metric)

    lower_pde, upper_pde = _get_metric_bounds(metric, df_pde)
    lower_mc, upper_mc = _get_metric_bounds(metric, df_mc)

    lower = lower_pde if lower_pde is not None else lower_mc
    upper = upper_pde if upper_pde is not None else upper_mc

    y_low_pde, y_high_pde = _clip_band(mean_pde, std_pde, lower, upper)
    y_low_mc, y_high_mc = _clip_band(mean_mc, std_mc, lower, upper)

    # compute common y-limits from BOTH panels
    combined_low = min(np.min(y_low_pde), np.min(y_low_mc))
    combined_high = max(np.max(y_high_pde), np.max(y_high_mc))

    if lower is not None:
        combined_low = max(combined_low, lower)
    if upper is not None:
        combined_high = min(combined_high, upper)

    # small padding
    pad = 0.03 * (combined_high - combined_low) if combined_high > combined_low else 0.05
    y_min = combined_low - pad if lower is None else max(lower, combined_low - pad)
    y_max = combined_high + pad if upper is None else min(upper, combined_high + pad)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # PDE panel
    ax = axes[0]
    ax.plot(x_pde, mean_pde, marker="o", linewidth=line_width, markersize=marker_size)
    ax.fill_between(x_pde, y_low_pde, y_high_pde, alpha=alpha_band)
    if panel_titles:
        ax.set_title(pde_panel_title, fontsize=panel_title_fontsize)
    ax.set_xlabel(pde_xlabel, fontsize=label_fontsize)
    ax.set_ylabel(metric_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_ylim(y_min, y_max)

    # MC panel
    ax = axes[1]
    ax.plot(x_mc, mean_mc, marker="o", linewidth=line_width, markersize=marker_size)
    ax.fill_between(x_mc, y_low_mc, y_high_mc, alpha=alpha_band)
    if panel_titles:
        ax.set_title(mc_panel_title, fontsize=panel_title_fontsize)
    ax.set_xlabel(mc_xlabel, fontsize=label_fontsize)
    # ax.set_ylabel(metric_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_ylim(y_min, y_max)

    if show_title:
        fig.suptitle(figure_title or metric_label, fontsize=title_fontsize)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, axes

def make_comparison_figures(
    pde_csv_path,
    mc_csv_path,
    out_dir,
    metrics=("bbox_density", "D_est", "R_g"),
    show_title=False,
    panel_titles=True,
    title_fontsize=18,
    panel_title_fontsize=14,
    label_fontsize=13,
    tick_fontsize=11,
    line_width=2.2,
    marker_size=6,
    alpha_band=0.20,
):
    os.makedirs(out_dir, exist_ok=True)

    df_pde = pd.read_csv(pde_csv_path)
    df_mc = pd.read_csv(mc_csv_path)

    for metric in metrics:
        if metric not in df_pde.columns or metric not in df_mc.columns:
            print(f"Skipping {metric}: not found in both CSVs.")
            continue

        out_path = os.path.join(out_dir, f"{metric}_comparison.png")

        plot_metric_comparison(
            metric=metric,
            df_pde=df_pde,
            df_mc=df_mc,
            out_path=out_path,
            metric_label=METRIC_LABELS.get(metric, metric),
            pde_panel_title="PDE DLA",
            mc_panel_title="Monte Carlo DLA",
            show_title=show_title,
            figure_title=METRIC_LABELS.get(metric, metric),
            panel_titles=panel_titles,
            title_fontsize=title_fontsize,
            panel_title_fontsize=panel_title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            line_width=line_width,
            marker_size=marker_size,
            alpha_band=alpha_band,
        )

    print("Saved comparison figures to:", os.path.abspath(out_dir))



if __name__ == "__main__":
    make_comparison_figures(
        pde_csv_path="../assignment_2/data/dla/pde_ita_metrics.csv",
        mc_csv_path="../assignment_2/data/dla_mc/mc_ps_metrics.csv",
        out_dir="../assignment_2/data/comparison_figures",
        metrics=("bbox_density", "D_est", "R_g"),
        show_title=False,
        panel_titles=False,
        title_fontsize=18,
        panel_title_fontsize=14,
        label_fontsize=13,
        tick_fontsize=11,
        line_width=2.2,
        marker_size=6,
        alpha_band=0.18,
    )

