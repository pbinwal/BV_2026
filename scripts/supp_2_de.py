"""supp_2.py

Compares z-scored acoustic distances between syllable pairs across correlation
categories (non-significant, positive, negative) for adjacent and next contexts.

Data sources
------------
  data/acoustic_distances/bird_{n}/bird_{n}_context_sensitive.csv
      Non-symmetric. Used when analysing acoustic distances between correlated
      phrases that were adjacent, to account for subtle differences in acoustics
      (Wohlgemuth, 2010).
  data/acoustic_distances/bird_{n}/bird_{n}_context_agnostic.csv
      Symmetric. Used for next phrase-pair analyses.
  figures/Figure 2/adjacent_piechart_ztest_pairs.csv
  figures/Figure 2/next_piechart_ztest_pairs.csv

Outputs (saved to figures/Supplementary/Supp_2/)
------------------------------------------------
  supp_2_a.png  — z-scored acoustic distance (x) vs correlation rho (y), adjacent
  supp_2_b.png  — z-scored acoustic distance (x) vs correlation rho (y), next

Intermediate CSVs saved to output/Acoustic distances z corrected/:
  combined_acoustic_corr_adj_df.csv
  combined_acoustic_corr_next_df.csv
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy.stats import kruskal, zscore

from get_repeat_data import REPO_ROOT, syllables_mapping

# ── Paths ─────────────────────────────────────────────────────────────────────
ACOUSTIC_DIR   = os.path.join(REPO_ROOT, "data", "acoustic_distances")
ADJACENT_CSV   = os.path.join(REPO_ROOT, "figures", "Figure 2", "adjacent_piechart_ztest_pairs.csv")
NEXT_CSV       = os.path.join(REPO_ROOT, "figures", "Figure 2", "next_piechart_ztest_pairs.csv")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "output", "Acoustic distances z corrected")
SUPP_SAVE_DIR  = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUPP_SAVE_DIR, exist_ok=True)

BIRD_NUMS = list(syllables_mapping.keys())   # ["1", "2", ..., "6"]

CATEGORY_COLORS = {
    "non-significant": "#bdbdbd",
    "positive":        "#e66557",
    "negative":        "#6d90f0",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_acoustic_distances(is_adjacent: bool) -> pd.DataFrame:
    """Load acoustic distance matrices for all birds; return long-format DataFrame."""
    rows = []
    for bird_num in BIRD_NUMS:
        bird_dir = os.path.join(ACOUSTIC_DIR, f"bird_{bird_num}")
        if is_adjacent:
            csv_path = os.path.join(bird_dir, f"bird_{bird_num}_context_sensitive.csv")
        else:
            csv_path = os.path.join(bird_dir, f"bird_{bird_num}_context_agnostic.csv")

        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)

        # Bird 3: acoustic files use 'y', BV data uses 'd'
        if bird_num == "3":
            df.index   = df.index.str.replace('y', 'd', regex=False)
            df.columns = df.columns.str.replace('y', 'd', regex=False)

        for row_syl in df.index:
            for col_syl in df.columns:
                rows.append({
                    "bird_num":      bird_num,
                    "syllable_pair": f"{row_syl}_{col_syl}",
                    "acoustic_dist": df.loc[row_syl, col_syl],
                })

    return pd.DataFrame(rows)


def process_data(acoustic_df: pd.DataFrame, corr_df: pd.DataFrame,
                 context_type: str) -> pd.DataFrame:
    """Merge acoustic distances with correlation categories; z-score per bird."""
    df = acoustic_df.dropna(subset=["acoustic_dist"]).copy()

    # Adjacent: drop self-pairs (a_a, b_b, …)
    if context_type == "adjacent":
        df = df[~df["syllable_pair"].str.match(r"^(.)_\1$")]

    # Z-score within each bird
    df["z_scored_acoustic_dist"] = df.groupby("bird_num")["acoustic_dist"].transform(
        lambda x: zscore(x.astype(float))
    )

    # Split pair into syl1/syl2 for merging
    df[["syl1", "syl2"]] = df["syllable_pair"].str.split("_", expand=True)
    df["bird_num"] = df["bird_num"].astype(str)

    corr_df = corr_df.copy()
    corr_df["bird_num"] = corr_df["bird_num"].astype(str)

    merged = pd.merge(
        df,
        corr_df[["bird_num", "syl1", "syl2", "rho", "pie_category"]],
        how="left",
        on=["bird_num", "syl1", "syl2"],
    ).rename(columns={"pie_category": "sig_status_aft_all_corr"})

    combined = merged[["bird_num", "syllable_pair", "acoustic_dist",
                        "z_scored_acoustic_dist", "rho", "sig_status_aft_all_corr"]]

    # Drop rows where significant categories have no rho
    combined = combined[
        ~((combined["sig_status_aft_all_corr"].isin(["positive", "negative"])) &
          (combined["rho"].isna()))
    ]
    combined = combined.dropna(subset=["sig_status_aft_all_corr"])
    return combined


def plot_scatter(combined_df: pd.DataFrame, context_type: str,
                 save_path: str) -> None:
    """Scatter of z-scored acoustic distance (x) vs correlation rho (y)."""
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plot_df = combined_df.dropna(
        subset=["z_scored_acoustic_dist", "rho", "sig_status_aft_all_corr"]
    ).copy()

    # For next context: link mirror pairs (a_b <-> b_a) within the same bird
    if context_type == "next":
        plot_df[["_s1", "_s2"]] = plot_df["syllable_pair"].str.split("_", expand=True)
        for _, row in plot_df.iterrows():
            mirror_pair = f"{row['_s2']}_{row['_s1']}"
            mirror = plot_df[
                (plot_df["syllable_pair"] == mirror_pair) &
                (plot_df["bird_num"] == row["bird_num"])
            ]
            if not mirror.empty:
                mirror_row = mirror.iloc[0]
                ax.plot(
                    [row["z_scored_acoustic_dist"], mirror_row["z_scored_acoustic_dist"]],
                    [row["rho"], mirror_row["rho"]],
                    color="darkgrey", linewidth=0.5, zorder=1,
                )
        plot_df = plot_df.drop(columns=["_s1", "_s2"])

    for cat in ["non-significant", "positive", "negative"]:
        subset = plot_df[plot_df["sig_status_aft_all_corr"] == cat]
        if subset.empty:
            continue
        ax.scatter(
            subset["z_scored_acoustic_dist"],
            subset["rho"],
            color=CATEGORY_COLORS[cat],
            s=20, alpha=0.6, edgecolor="black", linewidth=0.5, zorder=2,
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)

    ax.set_xlabel("Acoustic Distance\n(z-scored)", fontsize=10)
    ax.set_ylabel("Correlation", fontsize=10)
    ax.tick_params(axis="both", which="major", length=6, width=1, labelsize=10)
    ax.tick_params(axis="both", which="minor", length=3, width=0.5)

    ax.xaxis.set_major_locator(FixedLocator([-2.5, 0, 2.5]))
    ax.set_xlim(left=min(ax.get_xlim()[0], -2.7), right=max(ax.get_xlim()[1], 2.7))
    ax.xaxis.set_minor_locator(FixedLocator([-1.25, 1.25]))
    ax.yaxis.set_major_locator(FixedLocator([-0.8, 0, 0.8]))
    ax.set_ylim(bottom=min(ax.get_ylim()[0], -0.9), top=max(ax.get_ylim()[1], 0.9))
    ax.yaxis.set_minor_locator(FixedLocator([-0.4, 0.4]))

    ax.set_title("Adjacent" if context_type == "adjacent" else "Next", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def run_kruskal_wallis(combined_df: pd.DataFrame, context_type: str) -> None:
    groups = [
        grp["z_scored_acoustic_dist"].values
        for _, grp in combined_df.groupby("sig_status_aft_all_corr")
    ]
    stat, p = kruskal(*groups)
    print(f"\n{context_type.upper()} Kruskal-Wallis: H={stat:.3f}, p={p:.3g}")


def print_summary(combined_df: pd.DataFrame, context_type: str) -> None:
    print(f"\n[SUMMARY] {context_type.upper()} — z-scored acoustic distance by category:")
    for cat in combined_df["sig_status_aft_all_corr"].unique():
        vals = combined_df.loc[combined_df["sig_status_aft_all_corr"] == cat, "z_scored_acoustic_dist"]
        print(f"  {cat:>15}: mean={vals.mean():.3f}, median={vals.median():.3f}, n={len(vals)}")


# ── Main ──────────────────────────────────────────────────────────────────────

adj_corr_df  = pd.read_csv(ADJACENT_CSV)
next_corr_df = pd.read_csv(NEXT_CSV)

# ── Adjacent ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROCESSING ADJACENT CONTEXT")
print("=" * 60)

adj_acoustic_df = load_acoustic_distances(is_adjacent=True)
combined_adj     = process_data(adj_acoustic_df, adj_corr_df, context_type="adjacent")
combined_adj.to_csv(os.path.join(OUTPUT_DIR, "combined_acoustic_corr_adj_df.csv"), index=False)

plot_scatter(combined_adj, context_type="adjacent",
             save_path=os.path.join(SUPP_SAVE_DIR, "supp_2_a.png"))
run_kruskal_wallis(combined_adj, "adjacent")
print_summary(combined_adj, "adjacent")

# ── Next ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROCESSING NEXT CONTEXT")
print("=" * 60)

next_acoustic_df = load_acoustic_distances(is_adjacent=False)
combined_next     = process_data(next_acoustic_df, next_corr_df, context_type="next")
combined_next.to_csv(os.path.join(OUTPUT_DIR, "combined_acoustic_corr_next_df.csv"), index=False)

plot_scatter(combined_next, context_type="next",
             save_path=os.path.join(SUPP_SAVE_DIR, "supp_2_b.png"))
run_kruskal_wallis(combined_next, "next")
print_summary(combined_next, "next")
