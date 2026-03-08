import os
import pandas as pd

# Configure pandas to display all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def make_combined_shape_table(
    pde_csv_path,
    mc_csv_path,
    pde_big_csv_path=None,
    out_dir=None,
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
                "Parameter": param_val,
                "Model": model_name
            }
            for metric in metrics:
                m = g[metric].mean()
                s = g[metric].std()
                row[metric] = rf"${m:.{decimals}f} \pm {s:.{decimals}f}$"
            rows.append(row)
        return pd.DataFrame(rows)

    df_pde = pd.read_csv(pde_csv_path)
    df_mc = pd.read_csv(mc_csv_path)

    pde_table = summarize(df_pde, "ita", "PDE (N=100)")
    mc_table = summarize(df_mc, "ps", "MC (N=100)")

    # If pde_big_csv_path is provided, include those results
    if pde_big_csv_path is not None:
        df_pde_big = pd.read_csv(pde_big_csv_path)
        pde_big_table = summarize(df_pde_big, "ita", "PDE (N=300, opt)")

        # Remove Model column before merging to avoid suffix issues
        pde_table_for_merge = pde_table.drop(columns=["Model"])
        pde_big_table_for_merge = pde_big_table.drop(columns=["Model"])

        # Merge PDE tables so N=100 and N=300 results are side-by-side for each eta
        pde_merged = pde_table_for_merge.merge(
            pde_big_table_for_merge,
            on="Parameter",
            suffixes=("_N100", "_N300")
        )

        # Reorganize columns to alternate between N=100 and N=300 for each metric
        new_columns = ["Parameter"]
        for metric in metrics:
            new_columns.append(f"{metric}_N100")
            new_columns.append(f"{metric}_N300")

        pde_merged = pde_merged[new_columns]

        # Rename columns for clarity
        rename_dict = {}
        for metric in metrics:
            rename_dict[f"{metric}_N100"] = f"{metric} (N=100)"
            rename_dict[f"{metric}_N300"] = f"{metric} (N=300)"
        pde_merged = pde_merged.rename(columns=rename_dict)

        # Add model identifier column
        pde_merged.insert(1, "Type", "PDE")

        # Prepare MC table: rename columns to have (N=100) suffix and drop Model column
        mc_table_prepared = mc_table.drop(columns=["Model"])
        mc_rename_dict = {}
        for metric in metrics:
            mc_rename_dict[metric] = f"{metric} (N=100)"
        mc_table_prepared = mc_table_prepared.rename(columns=mc_rename_dict)
        mc_table_prepared.insert(1, "Type", "MC")

        combined = pd.concat([
            pde_merged,
            mc_table_prepared
        ], ignore_index=True)
    else:
        pde_table.insert(1, "Type", "PDE")
        mc_table.insert(1, "Type", "MC")
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
    pde_big_csv_path="../data/dla/pde_ita_metrics_big.csv",
    out_dir="../data/comparison_tables"
)
print(combined)