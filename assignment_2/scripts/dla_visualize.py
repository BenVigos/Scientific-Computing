# python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dla_metrics(csv_path=None, out_dir=None):
    # defaults (common locations)
    if csv_path is None:
        candidates = [
            os.path.join("..", "data", "dla", "dla_metrics.csv"),
            os.path.join("assignment_2", "data", "dla", "dla_metrics.csv"),
            os.path.join("data", "dla", "dla_metrics.csv"),
            "dla_metrics.csv",
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ita"])  # ensure ita present
    df["ita"] = df["ita"].astype(float)

    sns.set(style="whitegrid", context="talk")

    # 1) mean +/- std trends for key metrics
    metrics = ["D_est", "R_g", "occupancy"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4*len(metrics)), sharex=True)
    for ax, metric in zip(axes, metrics):
        # plot mean with shaded std
        grouped = df.groupby("ita")[metric].agg(["mean", "std", "count"]).reset_index()
        ax.plot(grouped["ita"], grouped["mean"], marker="o", label=f"{metric} mean")
        ax.fill_between(grouped["ita"],
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.3, label=f"{metric} ± std")
        ax.set_ylabel(metric)
        ax.legend()
    axes[-1].set_xlabel("ita")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_trends.png"), dpi=150)

    # 2) boxplot of fractal dimension per ita
    plt.figure(figsize=(8,5))
    sns.boxplot(x="ita", y="D_est", data=df, color="C0")
    sns.swarmplot(x="ita", y="D_est", data=df, color="k", alpha=0.6, size=3)
    plt.xlabel("ita")
    plt.ylabel("Estimated fractal dimension (D_est)")
    plt.title("Distribution of D_est per ita")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D_est_boxplot.png"), dpi=150)

    # 3) scatter D_est vs R_g colored by ita
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df, x="R_g", y="D_est", hue="ita", palette="viridis", s=60, edgecolor="w")
    plt.xlabel("Radius of gyration (R_g)")
    plt.ylabel("D_est")
    plt.title("D_est vs R_g (color = ita)")
    plt.legend(title="ita", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D_vs_Rg_scatter.png"), dpi=150)

    # 4) correlation heatmap for numeric metrics
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore")
    corr = numeric.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation matrix (numeric metrics)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_correlation.png"), dpi=150)

    print("Saved figures to:", os.path.abspath(out_dir))
    plt.show()


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_dla_metrics(csv_path=csv_arg)
