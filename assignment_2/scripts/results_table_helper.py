import os
import pandas as pd
import numpy as np


def make_combined_shape_table(
    pde_csv_path,
    mc_csv_path,
    out_dir,
    metrics=("max_width", "height", "aspect_ratio"),
    decimals=2,
    save_csv=True,
    save_latex=True,
):
    os.makedirs(out_dir, exist_ok=True)

    def summarize(df, param_col, model_name):
        df = df.copy()
        df[param_col] = pd.to_numeric(df[param_col], errors="coerce")
        df = df.dropna(subset=[param_col])

        if param_col == "ita":
            df[param_col] = df[param_col].round(1)

        rows = []
        for param_val, g in df.groupby(param_col):
            row = {
                "Model": model_name,
                "Parameter": param_val
            }
            for metric in metrics:
                m = g[metric].mean()
                s = g[metric].std()
                row[metric] = rf"${m:.{decimals}f} \pm {s:.{decimals}f}$"
            rows.append(row)
        return pd.DataFrame(rows)

    df_pde = pd.read_csv(pde_csv_path)
    df_mc = pd.read_csv(mc_csv_path)

    pde_table = summarize(df_pde, "ita", "PDE")
    mc_table = summarize(df_mc, "ps", "MC")

    combined = pd.concat([pde_table, mc_table], ignore_index=True)

    if save_csv:
        combined.to_csv(os.path.join(out_dir, "combined_shape_table.csv"), index=False)

    if save_latex:
        with open(os.path.join(out_dir, "combined_shape_table.tex"), "w") as f:
            f.write(combined.to_latex(index=False, escape=False))

    return combined

if __name__ == "__main__":
    combined = make_combined_shape_table(
    pde_csv_path="../data/dla/pde_ita_metrics.csv",
    mc_csv_path="../data/dla_mc/mc_ps_metrics.csv",
    out_dir="../data/comparison_tables"
)
print(combined)