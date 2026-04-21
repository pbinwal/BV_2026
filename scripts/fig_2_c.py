"""
Pie charts for adjacent and next correlation significance (Figure 2).

Workflow:
- Loads raw per-bird correlation CSVs.
- Applies Benjamini-Hochberg (FDR) correction to the raw p-values.
- For each BH-significant correlation, performs a z-test against a synthetic
  (null) distribution from 100 shuffled iterations.
- Applies a second BH correction to the z-test p-values.
- Categorizes each correlation as 'positive', 'negative', or 'non-significant'
  based on direction and whether it survives both corrections.
- Saves pie charts, histogram, and CSV pair-detail files to figures/Figure 2/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from get_repeat_data import syllables_mapping

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# Coolwarm-inspired colors matching other visualizations
CB_COLORS = {
    "non-significant": "#BDBBBB",  # grey
    "positive": "#f7af91ff",       # red (positive correlations)
    "negative": "#93b4e9ff",       # blue (negative correlations)
}


def autopct_format(pct, allvals):
    return f"{pct:.1f}%"


def apply_correction(df):
    """Apply Benjamini-Hochberg (FDR) correction."""
    df["p_value_bh"] = pd.NA
    df["new_significance_status"] = df["status"]

    valid_mask = (
        df["status"].isin(["significant", "nonsignificant"]) &
        df["p_value"].notna()
    )

    pvals = df.loc[valid_mask, "p_value"].values

    print(f"Total valid comparisons: {len(pvals)}")
    if len(pvals) == 0:
        print("WARNING: No valid p-values!")
        return df

    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    df.loc[valid_mask, "p_value_bh"] = pvals_corrected
    df.loc[valid_mask, "new_significance_status"] = [
        "significant" if r else "nonsignificant" for r in reject
    ]

    print("Counts BEFORE correction:")
    print(df["status"].value_counts())
    print("Counts AFTER correction (new_significance_status):")
    print(df["new_significance_status"].value_counts())

    return df


def calculate_z_test(observed_rho, synthetic_correlations):
    """
    One-sided z-test for observed correlation against synthetic distribution.
    Right-tailed if observed_rho > 0, left-tailed if observed_rho < 0.
    """
    synth_mean = np.mean(synthetic_correlations)
    synth_std = np.std(synthetic_correlations, ddof=1)

    if synth_std == 0:
        return np.nan, np.nan

    z_score = (observed_rho - synth_mean) / synth_std

    if observed_rho > 0:
        p_value = 1 - norm.cdf(z_score)
    else:
        p_value = norm.cdf(z_score)

    return z_score, p_value


def load_synthetic_correlations(synth_folder, bird_num, filename_suffix, n_iterations=100):
    """Load all synthetic correlation dataframes (one per iteration)."""
    dfs = []
    for i in range(1, n_iterations + 1):
        csv_path = os.path.join(
            REPO_ROOT, synth_folder,
            f"bird_{bird_num}",
            f"bird_{bird_num}_{i}{filename_suffix}"
        )
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"Warning: Missing synthetic file {csv_path}")

    return dfs if dfs else None


def analyze_correlations(corr_type="adjacent", alpha=0.05):
    """
    Analyze correlations for all birds:
    - Combine all observed correlations across birds
    - Apply first BH correction to original p-values
    - Run z-tests on BH-corrected significant correlations using synthetic data
    - Apply second BH correction to z-test p-values
    - Count how many are significant after both independent corrections

    Returns: (results dict, ztest_impact dict)
    """
    results = {"positive": 0, "negative": 0, "non-significant": 0}
    ztest_impact = {"remained_significant": 0, "lost_significance": 0}

    sig_positive = []
    sig_negative = []
    non_sig = []
    all_ztest_results = []

    yaml_dir = Path(REPO_ROOT) / "yamls"

    # Load all observed correlations
    all_observed_dfs = []
    for bird_num in syllables_mapping.keys():
        yaml_path = yaml_dir / f"bird_{bird_num}.yaml"
        if not yaml_path.is_file():
            continue
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}

        if corr_type == "adjacent":
            observed_dir = config.get("adjacent_dir")
            if not observed_dir:
                continue
            observed_file = os.path.join(REPO_ROOT, observed_dir, f"bird_{bird_num}_adjacent_corr_ALL.csv")
        else:
            observed_dir = config.get("next_dir")
            if not observed_dir:
                continue
            observed_file = os.path.join(REPO_ROOT, observed_dir, f"bird_{bird_num}_next_corr_ALL.csv")

        if not os.path.exists(observed_file):
            continue

        observed_df = pd.read_csv(observed_file)
        observed_df["bird_num"] = bird_num
        all_observed_dfs.append(observed_df)


    combined_df = pd.concat(all_observed_dfs, ignore_index=True) if all_observed_dfs else pd.DataFrame()

    if corr_type == "next":
        combined_df = combined_df.rename(columns={"correlation": "rho", "corrleation": "rho"})

    print(f"\nApplying BH correction...")
    combined_df = apply_correction(combined_df)

    # --- Save combined, BH-corrected DataFrame for downstream analysis (same as pie_charts.py) ---
    if not combined_df.empty:
        if corr_type == "adjacent":
            out_dir = os.path.join(REPO_ROOT, "output", "Adjacent Correlations")
            out_path = os.path.join(out_dir, "Adjacent_Correlations_correction_bh.csv")
        else:
            out_dir = os.path.join(REPO_ROOT, "output", "Next Correlations")
            out_path = os.path.join(out_dir, "Next_Correlations_correction_bh.csv")
        os.makedirs(out_dir, exist_ok=True)
        combined_df.to_csv(out_path, index=False)
        print(f"Saved combined BH-corrected correlations to: {out_path}")

    valid_mask = (
        combined_df["status"].isin(["significant", "nonsignificant"]) &
        (combined_df["new_significance_status"] != "insufficient_data")
    )
    total_pairs = valid_mask.sum()

    # Run z-tests for each bird
    for bird_num in syllables_mapping.keys():
        print(f"\nProcessing bird {bird_num}...")

        yaml_path = yaml_dir / f"bird_{bird_num}.yaml"
        if not yaml_path.is_file():
            continue
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}

        if corr_type == "adjacent":
            synth_dir = config.get("adjacent_dir_synth")
            filename_suffix = "_adjacent_corr_ALL_synth.csv"
        else:
            synth_dir = config.get("next_dir_synth")
            filename_suffix = "_next_corr_ALL_synth.csv"

        bird_df = combined_df[combined_df["bird_num"] == bird_num].copy()
        synth_dfs = load_synthetic_correlations(synth_dir, bird_num, filename_suffix)

        if synth_dfs is None:
            print(f"Warning: No synthetic data found for bird {bird_num}")
            continue

        if corr_type == "next":
            for synth_df in synth_dfs:
                if "correlation" in synth_df.columns:
                    synth_df.rename(columns={"correlation": "rho"}, inplace=True)

        corr_col = "rho"

        for idx, row in bird_df.iterrows():
            syl1 = row["syl1"]
            syl2 = row["syl2"]
            observed_rho = row[corr_col]
            n_samples = row["sample_size"]
            status = row["status"]
            new_status = row["new_significance_status"]

            if status not in ["significant", "nonsignificant"] or new_status == "insufficient_data":
                continue

            if new_status == "significant":
                synth_rho_values = []
                for synth_df in synth_dfs:
                    match = synth_df[(synth_df["syl1"] == syl1) & (synth_df["syl2"] == syl2)]
                    if not match.empty and match.iloc[0]["status"] != "insufficient_data":
                        synth_rho_values.append(match.iloc[0][corr_col])

                if len(synth_rho_values) > 0:
                    z_score, z_p_value = calculate_z_test(observed_rho, synth_rho_values)
                    if not pd.isna(z_p_value):
                        all_ztest_results.append({
                            "bird_num": bird_num,
                            "syl1": syl1,
                            "syl2": syl2,
                            "observed_rho": observed_rho,
                            "z_score": z_score,
                            "z_p_value": z_p_value,
                            "n_samples": n_samples,
                        })

    # Second BH correction on z-test p-values
    print(f"\nApplying BH correction to z-test p-values...")
    ztest_df = pd.DataFrame()
    if len(all_ztest_results) > 0:
        ztest_df = pd.DataFrame(all_ztest_results)
        z_pvals = ztest_df["z_p_value"].values
        reject, z_pvals_corrected, _, _ = multipletests(z_pvals, alpha=alpha, method="fdr_bh")
        ztest_df["z_p_value_bh"] = z_pvals_corrected
        ztest_df["z_significant"] = reject

        print(f"Total z-tests performed: {len(ztest_df)}")
        print(f"Z-tests significant after BH correction: {reject.sum()}")

        for idx, row in ztest_df.iterrows():
            corr_name = f"Bird {row['bird_num']}: {row['syl1']}→{row['syl2']} (ρ={row['observed_rho']:.3f})"
            if row["z_significant"]:
                ztest_impact["remained_significant"] += 1
                if row["observed_rho"] > 0:
                    results["positive"] += 1
                    sig_positive.append(corr_name)
                else:
                    results["negative"] += 1
                    sig_negative.append(corr_name)
            else:
                ztest_impact["lost_significance"] += 1
                results["non-significant"] += 1
                non_sig.append(corr_name)

    # Add first-BH-failed pairs to non-significant
    for idx, row in combined_df[valid_mask].iterrows():
        if row["new_significance_status"] != "significant":
            bird_num = row["bird_num"]
            syl1 = row["syl1"]
            syl2 = row["syl2"]
            observed_rho = row["rho"]
            corr_name = f"Bird {bird_num}: {syl1}→{syl2} (ρ={observed_rho:.3f})"
            if corr_name not in non_sig:
                results["non-significant"] += 1
                non_sig.append(corr_name)

    # Build and save pair-details CSV
    pair_details = []
    if not ztest_df.empty:
        for idx, row in ztest_df.iterrows():
            cat = ("positive" if row["observed_rho"] > 0 else "negative") if row["z_significant"] else "non-significant"
            pair_details.append({
                "bird_num": row["bird_num"],
                "syl1": row["syl1"],
                "syl2": row["syl2"],
                "rho": row["observed_rho"],
                "pie_category": cat,
            })
    for idx, row in combined_df[valid_mask].iterrows():
        if row["new_significance_status"] != "significant":
            pair_details.append({
                "bird_num": row["bird_num"],
                "syl1": row["syl1"],
                "syl2": row["syl2"],
                "rho": row["rho"],
                "pie_category": "non-significant",
            })

    pair_details_df = pd.DataFrame(pair_details)
    save_dir = os.path.join(REPO_ROOT, "figures", "Figure 2")
    os.makedirs(save_dir, exist_ok=True)
    base = "adjacent" if corr_type == "adjacent" else "next"
    pair_details_df.to_csv(os.path.join(save_dir, f"{base}_piechart_ztest_pairs.csv"), index=False)

    print(f"\nTotal valid pairs before correction: {total_pairs}")

    if sig_positive:
        print(f"\n{'='*60}")
        print(f"Z-TEST SIGNIFICANT POSITIVE ({len(sig_positive)}):")
        print(f"{'='*60}")
        for name in sig_positive:
            print(f"  {name}")

    if sig_negative:
        print(f"\n{'='*60}")
        print(f"Z-TEST SIGNIFICANT NEGATIVE ({len(sig_negative)}):")
        print(f"{'='*60}")
        for name in sig_negative:
            print(f"  {name}")

    if non_sig:
        print(f"\n{'='*60}")
        print(f"Z-TEST NON-SIGNIFICANT ({len(non_sig)}):")
        print(f"{'='*60}")
        for name in non_sig:
            print(f"  {name}")

    total_ztest = ztest_impact["remained_significant"] + ztest_impact["lost_significance"]
    if total_ztest > 0:
        print(f"\n{'='*60}")
        print(f"Z-TEST IMPACT ON BH-SIGNIFICANT CORRELATIONS:")
        print(f"{'='*60}")
        print(f"  Total BH-significant (z-tested): {total_ztest}")
        print(
            f"  Remained significant after z-test: {ztest_impact['remained_significant']} "
            f"({100*ztest_impact['remained_significant']/total_ztest:.1f}%)"
        )
        print(
            f"  Lost significance after z-test: {ztest_impact['lost_significance']} "
            f"({100*ztest_impact['lost_significance']/total_ztest:.1f}%)"
        )

    return results, ztest_impact


def create_ztest_impact_pie_chart(ztest_impact, title):
    """
    Prints z-test impact summary (remained vs lost significance after z-test).
    """
    sizes = [ztest_impact["remained_significant"], ztest_impact["lost_significance"]]
    total = sum(sizes)
    if total == 0:
        return

    print(f"\nZ-Test Impact Summary for {title}:")
    print(f"  Total BH-significant tested: {total}")
    print(f"  Remained significant: {sizes[0]} ({100*sizes[0]/total:.1f}%)")
    print(f"  Lost significance: {sizes[1]} ({100*sizes[1]/total:.1f}%)")


def create_pie_chart(results, title, save_path):
    """
    Pie chart for z-test results: positive / negative / non-significant.
    """
    base_labels = ["Positive", "Negative", "Non-significant"]
    sizes = [results["positive"], results["negative"], results["non-significant"]]
    colors = [CB_COLORS["positive"], CB_COLORS["negative"], CB_COLORS["non-significant"]]
    labels = [f"{lbl} ({cnt})" for lbl, cnt in zip(base_labels, sizes)]

    if sum(sizes) == 0:
        return

    fig = plt.figure(figsize=(1.3, 1.3), dpi=300)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: autopct_format(pct, sizes),
        textprops={"fontsize": 8},
    )
    ax.set_title(title, pad=10, fontsize=10)
    ax.set_aspect("equal")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(save_path.replace('.png', '.svg'), bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.show()
    plt.close()

    total = sum(sizes)
    print(f"\n{title} Summary:")
    print(f"  BH-corrected significant correlations tested: {total}")
    if results["positive"] > 0:
        print(f"  Z-Test Positive: {results['positive']} ({100*results['positive']/total:.1f}%)")
    if results["negative"] > 0:
        print(f"  Z-Test Negative: {results['negative']} ({100*results['negative']/total:.1f}%)")
    if results["non-significant"] > 0:
        print(f"  Z-Test Non-Significant: {results['non-significant']} ({100*results['non-significant']/total:.1f}%)")


def main():
    """
    Generate pie charts and histograms for Figure 2.
    Analyzes correlations significant after BH correction and shows how many
    remain significant after z-test.
    """
    print("Analyzing z-test results for BH-corrected significant correlations...")
    print("(Only correlations marked as 'significant' after first BH are z-tested)\n")

    alpha = 0.05

    print("\n" + "=" * 50)
    print("ADJACENT CORRELATIONS")
    print("=" * 50)
    adjacent_results, adjacent_ztest_impact = analyze_correlations(corr_type="adjacent", alpha=alpha)

    print("\n" + "=" * 50)
    print("NEXT CORRELATIONS")
    print("=" * 50)
    next_results, next_ztest_impact = analyze_correlations(corr_type="next", alpha=alpha)

    save_dir = os.path.join(REPO_ROOT, "figures", "Figure 2")
    os.makedirs(save_dir, exist_ok=True)

    # Individual z-test result pies
    adj_pie_path = os.path.join(save_dir, "adjacent_correlations_pie_z_test_corrected.png")
    next_pie_path = os.path.join(save_dir, "next_correlations_pie_z_test_corrected.png")
    create_pie_chart(adjacent_results, "Adjacent", adj_pie_path)
    create_pie_chart(next_results, "Next", next_pie_path)

    # Z-test impact pies
    print("\n" + "=" * 50)
    print("Z-Test Impact Summary")
    print("=" * 50)
    create_ztest_impact_pie_chart(adjacent_ztest_impact, "Adjacent Correlations\nZ-Test Impact")
    create_ztest_impact_pie_chart(next_ztest_impact, "Next Correlations\nZ-Test Impact")

    # Load pair-detail CSVs for downstream use
    adj_csv = os.path.join(save_dir, "adjacent_piechart_ztest_pairs.csv")
    next_csv = os.path.join(save_dir, "next_piechart_ztest_pairs.csv")

    adj_df = pd.read_csv(adj_csv) if os.path.exists(adj_csv) else pd.DataFrame()
    next_df = pd.read_csv(next_csv) if os.path.exists(next_csv) else pd.DataFrame()

    adj_pos_corr_df = adj_df[adj_df["pie_category"] == "positive"][["bird_num", "syl1", "syl2", "rho"]] if not adj_df.empty else pd.DataFrame()
    adj_neg_corr_df = adj_df[adj_df["pie_category"] == "negative"][["bird_num", "syl1", "syl2", "rho"]] if not adj_df.empty else pd.DataFrame()
    next_pos_corr_df = next_df[next_df["pie_category"] == "positive"][["bird_num", "syl1", "syl2", "rho"]] if not next_df.empty else pd.DataFrame()
    next_neg_corr_df = next_df[next_df["pie_category"] == "negative"][["bird_num", "syl1", "syl2", "rho"]] if not next_df.empty else pd.DataFrame()

    if not adj_df.empty:
        print("\nADJACENT POSITIVE CORRELATIONS:")
        print(adj_pos_corr_df)
        print(f"Number of rows: {len(adj_pos_corr_df)}")
        print("\nADJACENT NEGATIVE CORRELATIONS:")
        print(adj_neg_corr_df)
        print(f"Number of rows: {len(adj_neg_corr_df)}")

    if not next_df.empty:
        print("\nNEXT POSITIVE CORRELATIONS:")
        print(next_pos_corr_df)
        print(f"Number of rows: {len(next_pos_corr_df)}")
        print("\nNEXT NEGATIVE CORRELATIONS:")
        print(next_neg_corr_df)
        print(f"Number of rows: {len(next_neg_corr_df)}")

    # Summary table: mean ± SE
    print("\n" + "=" * 60)
    print("SUMMARY: Mean ± SE of Significant Correlations")
    print("=" * 60)
    print(f"{'Type':<20} {'Direction':<12} {'N':<6} {'Mean ρ':<10} {'SE':<10}")
    print("-" * 60)
    for corr_type, pos_df, neg_df in [
        ("Adjacent", adj_pos_corr_df, adj_neg_corr_df),
        ("Next", next_pos_corr_df, next_neg_corr_df),
    ]:
        for direction, df in [("Positive", pos_df), ("Negative", neg_df)]:
            if len(df) > 0:
                mean_rho = df["rho"].mean()
                se_rho = df["rho"].std(ddof=1) / np.sqrt(len(df))
                print(f"{corr_type:<20} {direction:<12} {len(df):<6} {mean_rho:<10.3f} {se_rho:<10.3f}")
            else:
                print(f"{corr_type:<20} {direction:<12} {'0':<6} {'N/A':<10} {'N/A':<10}")
    print("=" * 60)

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)

    # -----------------------------------------------------------------------
    # Histogram: distribution of rho values by category
    # -----------------------------------------------------------------------
    adj_nonsig_df = adj_df[adj_df["pie_category"] == "non-significant"].dropna(subset=["rho"]) if not adj_df.empty else pd.DataFrame()
    next_nonsig_df = next_df[next_df["pie_category"] == "non-significant"].dropna(subset=["rho"]) if not next_df.empty else pd.DataFrame()

    fig, axes = plt.subplots(2, 1, figsize=(2, 2.6), sharey=False)
    for ax, (corr_type, pos_df, neg_df, nonsig_df) in zip(axes, [
        ("Adjacent", adj_pos_corr_df, adj_neg_corr_df, adj_nonsig_df),
        ("Next", next_pos_corr_df, next_neg_corr_df, next_nonsig_df),
    ]):
        bins = np.arange(-0.8, 0.81, 0.1)
        data = [
            nonsig_df["rho"].dropna().values if not nonsig_df.empty else np.array([]),
            pos_df["rho"].dropna().values if not pos_df.empty else np.array([]),
            neg_df["rho"].dropna().values if not neg_df.empty else np.array([]),
        ]
        ax.hist(
            data, bins=bins,
            color=[CB_COLORS["non-significant"], CB_COLORS["positive"], CB_COLORS["negative"]],
            edgecolor="black", linewidth=0.4, label=["n.s.", "pos.", "neg."],
            alpha=0.85, stacked=True,
        )
        ax.set_xlabel("Correlation", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(corr_type, fontsize=11)
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xticks([-0.8, 0, 0.8])
        ax.set_xticklabels(["-0.8", "0", "+0.8"])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.4))
        if corr_type == "Adjacent":
            ax.set_yticks([0, 6, 12])
            ax.yaxis.set_minor_locator(plt.MultipleLocator(3))
        else:
            ax.set_yticks([0, 16, 32])
            ax.yaxis.set_minor_locator(plt.MultipleLocator(8))
        ax.tick_params(axis="both", which="major", labelsize=11, colors="black", direction="out", length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False, colors="black", direction="out", length=2, width=0.6)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(0.6)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels_leg, loc="lower center", ncol=3, fontsize=8,
        handlelength=1, handletextpad=0.4, borderpad=0.4,
        frameon=False, bbox_to_anchor=(0.5, -0.05),
    )
    for handle in leg.legend_handles:
        handle.set_edgecolor("none")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_2_c.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "fig_2_c.svg"), bbox_inches="tight")
    plt.show()

    # Normality tests
    from scipy.stats import shapiro
    for corr_type, pos_df, neg_df, nonsig_df in [
        ("Adjacent", adj_pos_corr_df, adj_neg_corr_df, adj_nonsig_df),
        ("Next", next_pos_corr_df, next_neg_corr_df, next_nonsig_df),
    ]:
        all_rho = pd.concat([
            pos_df["rho"] if not pos_df.empty else pd.Series(dtype=float),
            neg_df["rho"] if not neg_df.empty else pd.Series(dtype=float),
            nonsig_df["rho"] if not nonsig_df.empty else pd.Series(dtype=float),
        ]).dropna().values
        if len(all_rho) >= 3:
            stat_sw, p_sw = shapiro(all_rho)
            print(f"\n{corr_type} normality tests (n={len(all_rho)}):")
            print(f"  Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4e}")


if __name__ == "__main__":
    main()
