"""fig_1_b.py

Run for bird 1 to reproduce Fig 1: B

Plots per-syllable repeat-number distributions for a single bird (Fig 1B).
One panel per syllable, all sharing the same x- and y-axes.
Colours follow the Okabe-Ito palette (same as RA).

Output
------
  figures/Figure 1/fig_1_b.png
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from get_repeat_data import REPO_ROOT, syllables_mapping
from get_compressed_syntax import load_and_process_data

# ── Colour palette (Okabe-Ito, matching RA) ───────────────────────────────────
SYLLABLE_COLORS = {
    'a': '#009e73ff',
    'b': '#d55e00ff',
    'e': '#e69f00ff',
    'g': '#cc79a7ff',
    'h': '#0072b2ff',
    'u': '#56b4e9ff',
}

SAVE_DIR = Path(REPO_ROOT) / "figures" / "Figure 1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_syllable_repeats(df, syllable):
    repeats = []
    for song in df['compressed_song']:
        repeats.extend(int(m) for m in re.findall(rf'{syllable}(\d+)', song))
    return repeats


def plot_panel(repeats_list, syllable, ax, global_xlim, yticks, bird_num):
    color = SYLLABLE_COLORS.get(syllable, '#333333')
    bin_max = 38 if bird_num == "5" else (max(repeats_list) if repeats_list else 23)
    bins = np.arange(-0.5, bin_max + 1.5, 1)
    ax.hist(repeats_list, bins=bins, color=color, edgecolor='black',
            alpha=0.7, density=True)
    ax.set_xlim(*global_xlim)
    ax.set_ylim(0, 0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x ticks: major every 4, minor every 2
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))

    # y ticks from yticks arg
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.125))

    ax.tick_params(axis='both', which='major', width=1.2, length=4, labelsize=9)
    ax.tick_params(axis='both', which='minor', width=1.0, length=2)

    # clear labels (for Inkscape editing); title shows syllable letter
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(syllable.upper(), fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bird_num = input("Enter bird number (1-6): ").strip()
    if bird_num not in syllables_mapping:
        print(f"Invalid bird number: {bird_num}.")
        sys.exit(1)

    unique_syllables = syllables_mapping[bird_num]
    df = load_and_process_data(bird_num)

    syllable_distributions = {
        syl: get_syllable_repeats(df, syl) for syl in unique_syllables
    }

    # Global x-range shared across all panels
    all_repeats = [r for sub in syllable_distributions.values() for r in sub]
    if bird_num == "5":
        global_xlim = (0, 38)
    else:
        global_xlim = (0, (max(all_repeats) if all_repeats else 10) + 1)

    yticks = np.arange(0, 0.76, 0.25)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    num_syl = len(unique_syllables)
    fig, axes = plt.subplots(1, num_syl, figsize=(num_syl * 1.45, 1.1), sharey=True)
    if num_syl == 1:
        axes = [axes]

    for idx, syl in enumerate(unique_syllables):
        plot_panel(syllable_distributions[syl], syl, axes[idx],
                   global_xlim, yticks, bird_num)

    plt.tight_layout(pad=0)
    plt.subplots_adjust(right=0.98, wspace=0.7)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "fig_1_b.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"[INFO] Saved: {save_path}")
    plt.show()
