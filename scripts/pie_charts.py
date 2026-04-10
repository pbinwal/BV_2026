import pandas as pd
import matplotlib.pyplot as plt
import glob

import yaml
from get_repeat_data import syllables_mapping
import os

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# setting one global plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# colour-blind people friendly colours :)
# Using coolwarm-inspired colors matching heatmaps in plot_adjacent_corr.py
CB_COLORS = {
    "non-significant": "#A8A8A8",    # grey
    "insufficient data": "#E69F00",  # orange (kept for compatibility)
    "positive": "#f7af91ff",           # red (positive correlations, like heatmap)
    "negative": "#93b4e9ff",           # blue (negative correlations, like heatmap)
}



# Enter choice of correction method
# Use Benjamini–Hochberg (FDR) correction

# helper functions
def read_and_combine(file_patterns):
    # collect files 
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No files found for patterns: {file_patterns}")
    df_list = [pd.read_csv(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def autopct_format(pct, allvals):
    # Only show percentage inside the slice
    return f"{pct:.1f}%"

from statsmodels.stats.multitest import multipletests

def apply_correction(df):
    """Apply Benjamini-Hochberg (FDR) correction using statsmodels."""
    
    # Initialize columns
    df["p_value_bh"] = pd.NA
    df["new_significance_status"] = df["status"]  # keep original as base

    # Only apply to rows that are significant/nonsignificant and have p_value
    valid_mask = (
        df["status"].isin(["significant", "nonsignificant"]) &
        df["p_value"].notna()
    )

    pvals = df.loc[valid_mask, "p_value"].values
    
    print(f"\nTotal valid comparisons: {len(pvals)}")
    if len(pvals) == 0:
        print("WARNING: No valid p-values!")
        return df

    # Apply BH correction
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    # Assign corrected p-values
    df.loc[valid_mask, "p_value_bh"] = pvals_corrected
    # Assign new significance status
    df.loc[valid_mask, "new_significance_status"] = ["significant" if r else "nonsignificant" for r in reject]

    print("Counts BEFORE correction:")
    print(df["status"].value_counts())
    print("Counts AFTER correction (new_significance_status):")
    print(df["new_significance_status"].value_counts())

    return df


def plot_combined_pie(df, ax, title):
    def categorize(row):
        if row["new_significance_status"] == "insufficient_data":
            return "insufficient data"
        elif row["new_significance_status"] != "significant":
            return "non-significant"
        else:
            return "positive" if row["rho"] > 0 else "negative"

    combined = df.apply(categorize, axis=1)
    counts = combined.value_counts()
    counts = counts[counts.index.isin(CB_COLORS.keys())]

    # Print out which pairs are counted in each category
    print(f"\nPie chart for: {title}")
    for cat in ["positive", "negative", "non-significant"]:
        print(f"\n{cat.upper()} pairs:")
        for idx, row in df[combined == cat].iterrows():
            print(f"  {row['bird_num']}: {row['syl1']}→{row['syl2']} (rho={row['rho']}, status={row['new_significance_status']})")
        print(f"Total {cat}: {sum(combined == cat)}")

    if counts.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=24)
        ax.set_title(title)
        return

    colors = [CB_COLORS[c] for c in counts.index]
    # Add raw counts to labels (outside)
    labels_with_counts = [f"{cat} ({counts[cat]})" for cat in counts.index]
    ax.pie(
        counts,
        labels=labels_with_counts,
        colors=colors,
        autopct=lambda pct: autopct_format(pct, counts),
        textprops={"fontsize": 24}
    )
    ax.set_title(title, pad=22)



# Saving with suffix bh to denote Benjamini–Hochberg correction
suffix = "_bh"

# BUILD FILE PATTERNS FIRST
yaml_dir = os.path.join(REPO_ROOT, "yamls")
adjacent_files = []
next_files = []
save_adjacent_dir = None
save_next_dir = None

for bird_num in syllables_mapping.keys():
    yaml_path = os.path.join(yaml_dir, f"bird_{bird_num}.yaml")
    if not os.path.isfile(yaml_path):
        continue
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    adj_dir = cfg.get("adjacent_dir")
    if adj_dir:
        adj_dir_abs = os.path.join(REPO_ROOT, adj_dir)
        adjacent_files.append(os.path.join(adj_dir_abs, f"bird_{bird_num}_adjacent_corr_ALL.csv"))
        if save_adjacent_dir is None:
            save_adjacent_dir = adj_dir_abs

    next_dir = cfg.get("next_dir")
    if next_dir:
        next_dir_abs = os.path.join(REPO_ROOT, next_dir)
        next_files.append(os.path.join(next_dir_abs, f"bird_{bird_num}_next_corr_ALL.csv"))
        if save_next_dir is None:
            save_next_dir = next_dir_abs

# now read data - with bird_num tracking
df_adjacent_list = []
df_next_list = []

for bird_num in syllables_mapping.keys():
    yaml_path = os.path.join(yaml_dir, f"bird_{bird_num}.yaml")
    if not os.path.isfile(yaml_path):
        continue
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    adj_dir = cfg.get("adjacent_dir")
    if adj_dir:
        adj_file = os.path.join(REPO_ROOT, adj_dir, f"bird_{bird_num}_adjacent_corr_ALL.csv")
        if os.path.exists(adj_file):
            df = pd.read_csv(adj_file)
            df['bird_num'] = bird_num
            df_adjacent_list.append(df)

    next_dir = cfg.get("next_dir")
    if next_dir:
        next_file = os.path.join(REPO_ROOT, next_dir, f"bird_{bird_num}_next_corr_ALL.csv")
        if os.path.exists(next_file):
            df = pd.read_csv(next_file)
            df['bird_num'] = bird_num
            df_next_list.append(df)

df_adjacent = pd.concat(df_adjacent_list, ignore_index=True) if df_adjacent_list else pd.DataFrame()
df_next = pd.concat(df_next_list, ignore_index=True) if df_next_list else pd.DataFrame()
df_next = df_next.rename(columns={"correlation": "rho", "corrleation": "rho"})

# Apply BH correction
df_adjacent = apply_correction(df_adjacent)
df_next = apply_correction(df_next)

# Ensure directories exist
os.makedirs(save_adjacent_dir, exist_ok=True)
os.makedirs(save_next_dir, exist_ok=True)

# Build output file paths
output_adj_path = os.path.join(save_adjacent_dir, f"Adjacent_Correlations_correction{suffix}.csv")
output_next_path = os.path.join(save_next_dir, f"Next_Correlations_correction{suffix}.csv")

# Save combined CSVs
df_adjacent.to_csv(output_adj_path, index=False)
df_next.to_csv(output_next_path, index=False)


# --- Save pairwise pie chart assignments for debugging/comparison ---
def get_piechart_pairs(df, pie_type):
    # Only include pairs with status significant or nonsignificant (not insufficient_data)
    filtered = df[df["new_significance_status"] != "insufficient_data"].copy()
    def pie_category(row):
        # Insufficient data if sample size < 100
        if "n" in row and pd.notna(row["n"]) and row["n"] < 100:
            return "insufficient_data"
        # Non-significant if not significant
        if row["new_significance_status"] != "significant":
            return "non-significant"
        # Otherwise, positive/negative based on rho
        return "positive" if row["rho"] > 0 else "negative"
    filtered["pie_category"] = filtered.apply(pie_category, axis=1)
    # Ensure columns exist
    for col in ["bird_num", "syl1", "syl2", "rho", "status", "new_significance_status", "pie_category"]:
        if col not in filtered.columns:
            filtered[col] = pd.NA
    # Only keep relevant columns
    filtered = filtered[["bird_num", "syl1", "syl2", "rho", "status", "new_significance_status", "pie_category"]]
    return filtered


# Save pairwise CSVs for adjacent and next pies in Figure 2 directory
figure2_dir = os.path.join(REPO_ROOT, "figures", "Figure 2")
os.makedirs(figure2_dir, exist_ok=True)
adj_pairs = get_piechart_pairs(df_adjacent, "adjacent")
next_pairs = get_piechart_pairs(df_next, "next")
adj_pairs_path = os.path.join(figure2_dir, "DEBUG_pie_adj.csv")
next_pairs_path = os.path.join(figure2_dir, "DEBUG_pie_next.csv")
print(f"Saving adjacent pairs to: {adj_pairs_path}")
print(f"Saving next pairs to: {next_pairs_path}")
adj_pairs.to_csv(adj_pairs_path, index=False)
next_pairs.to_csv(next_pairs_path, index=False)

# Plotting with insufficient data
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
plot_combined_pie(df_adjacent, axes[0], "A → B (Adjacent)")
plot_combined_pie(df_next, axes[1], "A → Next repeat of B")
plt.tight_layout()
plt.show()


########### without insufficient
def plot_combined_pie_no_insufficient(df, ax, title):
    """
    Plot a pie chart of correlations, excluding 'insufficient_data'.
    Significant correlations are split into positive and negative,
    while non-significant correlations are grouped together.
    """
    # Exclude insufficient data
    df = df[df["new_significance_status"] != "insufficient_data"]

    if df.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=22)
        ax.set_title(title)
        return

    # Categorize each row
    def categorize(row):
        if row["new_significance_status"] != "significant":
            return "non-significant"
        else:
            return "positive" if row["rho"] > 0 else "negative"

    combined = df.apply(categorize, axis=1)
    counts = combined.value_counts()

    # Keep only categories present in CB_COLORS
    counts = counts[counts.index.isin(["non-significant", "positive", "negative"])]

    if counts.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=22)
        ax.set_title(title)
        return

    # Assign colors
    colors = [CB_COLORS[c] for c in counts.index]

    # Add raw counts to labels (outside)
    labels_with_counts = [f"{cat} ({counts[cat]})" for cat in counts.index]
    ax.pie(
        counts,
        labels=labels_with_counts,
        colors=colors,
        autopct=lambda pct: autopct_format(pct, counts),
        textprops={"fontsize": 8}
    )
    ax.set_title(title, pad=10, fontsize=10)



# Call the function
save_dir = os.path.join(REPO_ROOT, "figures", "Figure 2")
os.makedirs(save_dir, exist_ok=True)

fig1, ax1 = plt.subplots(1, 1, figsize=(1.25, 1.25), dpi=300)
plot_combined_pie_no_insufficient(df_adjacent, ax1, "Adjacent")
plt.tight_layout(pad=0)
fig1.savefig(os.path.join(save_dir, "adjacent_correlations_pie.png"), dpi=300, bbox_inches='tight', pad_inches=0)
fig1.savefig(os.path.join(save_dir, "adjacent_correlations_pie.svg"), dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

fig2, ax2 = plt.subplots(1, 1, figsize=(1.25, 1.25), dpi=300)
plot_combined_pie_no_insufficient(df_next, ax2, "Next")
plt.tight_layout(pad=0)
fig2.savefig(os.path.join(save_dir, "next_correlations_pie.png"), dpi=300, bbox_inches='tight', pad_inches=0)
fig2.savefig(os.path.join(save_dir, "next_correlations_pie.svg"), dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
