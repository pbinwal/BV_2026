"""
Significance pie charts for adjacent and next repeat contexts.
Performs k-sample Anderson-Darling tests (one per repeat phrase type per bird) asking
whether repeat distributions differ across different preceding/upcoming contexts.
Applies Benjamini-Hochberg correction across all tests combined, then plots
pies.
"""

import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import anderson_ksamp
from statsmodels.stats.multitest import multipletests

from get_compressed_syntax import load_and_process_data
from get_repeat_data import syllables_mapping

base_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(base_dir)
trans_dir = os.path.join(base_dir, "output", "Transition matrices")


def get_rpts_preceding(df, syl1, syl2):
    """Repeat counts of syl1 (the preceding syllable) when followed by syl2."""
    pattern = rf'{syl1}(\d+){syl2}\d+'
    counts = []
    for song in df['compressed_song']:
        counts.extend(int(m) for m in re.findall(pattern, song, flags=re.IGNORECASE))
    return counts


def get_rpts_upcoming(df, syl1, syl2):
    """Repeat counts of syl2 (the upcoming syllable) when preceded by syl1."""
    pattern = rf'(?:{syl1}\d+){syl2}(\d+)'
    counts = []
    for song in df['compressed_song']:
        counts.extend(int(m) for m in re.findall(pattern, song, flags=re.IGNORECASE))
    return counts


# Target syllables per bird — must match OG plot_rpts_by_context.py exactly
TARGET_SYLLABLES = {
    "1": ['a', 'u', 'g', 'h', 'e', 'b'],
    "2": ['b', 'c', 'e'],
    "3": ['b', 'c', 'd', 'e'],
    "4": ['b', 'e', 'k'],
    "5": ['b', 'e', 'f'],
    "6": ['b', 'e', 'h', 'm'],
}


# Collect one p-value per (bird, syllable) for each context direction,
# matching the OG script: only include a syllable if >= 2 pairs pass the
# transition-probability filter (trans_prob > 0.05), same as the valid_pairs
# filter used before writing the EMD CSVs in plot_rpts_by_context.py.
TRANS_THRESH = 0.05

all_preceding_pvals = []
all_upcoming_pvals = []

for bird_num, syllables in syllables_mapping.items():
    print(f"\nProcessing bird {bird_num}...")
    df = load_and_process_data(bird_num)
    target_syls = TARGET_SYLLABLES[bird_num]

    # Load transition matrix for this bird
    probs_file = os.path.join(trans_dir, f"bird_{bird_num}_transition_probs.npy")
    syll_file  = os.path.join(trans_dir, f"bird_{bird_num}_unique_syllables.npy")
    trans_probs     = np.load(probs_file)
    unique_syls_all = np.load(syll_file).tolist()
    syl_index = {s.upper(): i for i, s in enumerate(unique_syls_all)}

    # Build repeat lists for ALL syllable pairs from the transition matrix
    # (matches OG behaviour — includes non-target syllables like 'i', 'a', 'f')
    all_syls = [s.lower() for s in unique_syls_all]
    repeats_prec = {}
    repeats_upco = {}
    for syl1, syl2 in itertools.permutations(all_syls, 2):
        key = f"{syl1}{syl2}"
        # Preceding: syl2 is the repeating syllable → need syl2's count (≡ OG _aft)
        repeats_prec[key] = get_rpts_upcoming(df, syl1, syl2)
        # Upcoming:  syl1 is the repeating syllable → need syl1's count (≡ OG _bef)
        repeats_upco[key] = get_rpts_preceding(df, syl1, syl2)

    # Step 1: run k-sample AD on ALL non-empty pairs (no transition filter) -- matches OG
    ad_pvals_prec = {}
    for syl in target_syls:
        samples = [rpts for key, rpts in repeats_prec.items() if key[1] == syl and rpts]
        if len(samples) >= 2:
            try:
                result = anderson_ksamp(samples)
                ad_pvals_prec[syl] = result.pvalue
            except Exception:
                pass

    ad_pvals_upco = {}
    for syl in target_syls:
        samples = [rpts for key, rpts in repeats_upco.items() if key[0] == syl and rpts]
        if len(samples) >= 2:
            try:
                result = anderson_ksamp(samples)
                ad_pvals_upco[syl] = result.pvalue
            except Exception:
                pass

    # Step 2: only collect p-values for syllables that have >=2 pairs AFTER transition filter
    # (matches which syllables appear in the EMD CSVs that plot_emd_distributions.py reads)
    for syl in target_syls:
        filtered = [
            rpts for key, rpts in repeats_prec.items()
            if key[1] == syl and rpts
            and trans_probs[syl_index[key[0].upper()], syl_index[key[1].upper()]] > TRANS_THRESH
        ]
        if len(filtered) >= 2 and syl in ad_pvals_prec:
            all_preceding_pvals.append(ad_pvals_prec[syl])
            print(f"  Preceding bird {bird_num} syl={syl}: p={ad_pvals_prec[syl]:.4g}")

    for syl in target_syls:
        filtered = [
            rpts for key, rpts in repeats_upco.items()
            if key[0] == syl and rpts
            and trans_probs[syl_index[key[0].upper()], syl_index[key[1].upper()]] > TRANS_THRESH
        ]
        if len(filtered) >= 2 and syl in ad_pvals_upco:
            all_upcoming_pvals.append(ad_pvals_upco[syl])
            print(f"  Upcoming  bird {bird_num} syl={syl}: p={ad_pvals_upco[syl]:.4g}")

print(f"\nTotal tests: preceding={len(all_preceding_pvals)}, upcoming={len(all_upcoming_pvals)}")

# BH correction across all tests combined
all_pvals = all_preceding_pvals + all_upcoming_pvals
n_prec = len(all_preceding_pvals)

if all_pvals:
    reject, _, _, _ = multipletests(all_pvals, alpha=0.05, method='fdr_bh')
    reject_prec = reject[:n_prec]
    reject_upco = reject[n_prec:]
    sig_prec   = int(reject_prec.sum())
    nonsig_prec = n_prec - sig_prec
    sig_upco   = int(reject_upco.sum())
    nonsig_upco = len(all_upcoming_pvals) - sig_upco
else:
    sig_prec = nonsig_prec = sig_upco = nonsig_upco = 0

print(f"\nAfter BH correction:")
print(f"  Preceding context : {sig_prec} significant, {nonsig_prec} non-significant")
print(f"  Upcoming context  : {sig_upco} significant, {nonsig_upco} non-significant")

# Greyscale colours: [non-significant, significant]
grey_colors = ['#ffffff', '#bbbbbb']

# Create figure with two pies stacked vertically
fig_pies, (ax_pie1, ax_pie2) = plt.subplots(2, 1, figsize=(3.2, 8.8))

def draw_pie(ax, nonsig, sig, title):
    sizes = [nonsig, sig]
    labels = ['non-significant', 'significant']
    colors = grey_colors

    filtered_data = [
        (size, label, color)
        for size, label, color in zip(sizes, labels, colors)
        if size > 0
    ]
    if not filtered_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11)
        ax.set_title(title, fontsize=11)
        return

    f_sizes, f_labels, f_colors = zip(*filtered_data)

    wedges, _ = ax.pie(
        f_sizes,
        labels=None,
        colors=f_colors,
        startangle=90,
        radius=0.95,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.3},
    )

    # Place percentage labels outside each wedge
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(ang)) * 1.2
        y = np.sin(np.deg2rad(ang)) * 1.2
        ax.text(
            x, y,
            f"{100.0 * f_sizes[i] / sum(f_sizes):.1f}%",
            ha='center', va='center', fontsize=11,
        )

    ax.set_title(title, fontsize=11, pad=6)


draw_pie(ax_pie1, nonsig_upco, sig_upco, 'Upcoming Context')
draw_pie(ax_pie2, nonsig_prec, sig_prec, 'Preceding Context')


plt.tight_layout()

# Shared rectangle legend below both pies — always show both categories
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=grey_colors[0], edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, facecolor=grey_colors[1], edgecolor='black'),
]
legend_labels = ['non-sig.', 'sig.']
fig_pies.legend(
    legend_handles, legend_labels,
    fontsize=11, loc='lower center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=2, frameon=False,
)

# Save
save_dir = Path(repo_root) / "figures" / "Figure 1"
save_dir.mkdir(parents=True, exist_ok=True)

png_path = save_dir / "fig_1_g.png"

try:
    fig_pies.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Pie charts saved:\n  {png_path}")
except Exception as e:
    print(f"\n[ERROR] Failed to save figure: {e}")

plt.show()
