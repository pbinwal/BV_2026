# input syllable of interest and plot the distribution of repeat numbers

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from itertools import permutations
import re
from collections import defaultdict
import yaml
from pathlib import Path

from get_compressed_syntax import load_and_process_data
from get_repeat_data import syllables_mapping


def get_syllable_repeats(df, syllable):
    """
    This function finds and gets the repeat numbers for all repeats of a syllable in the data frame
    -Inputs
        df: data frame containing a compressed_song column containing compressed form of the song syntax. eg. aaabbuuuu becomes a3b1u4
        syllable: the syllable whose repeat number you are interested in
    -Returns
        repeats_list: a list containing all repeat numbers of a syllable in the data frame in serial order
    """
    repeats_list = []
    for song in df['compressed_song']:
        matches = re.findall(rf'{syllable}(\d+)', song)
        repeats = [int(match) for match in matches]
        repeats_list.extend(repeats)
    return repeats_list


def plot_repeat_distribution_relative(repeats_list, syllable, ax, bird_num="1"):
    """
    Plots the relative frequency distribution of repeat numbers for a given syllable
    on the specified axis.
    """
    if bird_num == "5":
        bin_max = 38
    else:
        bin_max = max(repeats_list) if repeats_list else 23
    bins = np.arange(0 - 0.5, bin_max + 1.5, 1)
    ax.hist(repeats_list, bins=bins, color = "#e69f00ff", edgecolor='black', alpha=0.7, density=True)
    ax.set_title(f"{syllable.upper()}", fontsize=18)
    ax.set_xlabel('Repeat Number', fontsize=18)
    ax.set_ylabel('Relative Frequency', fontsize=18)
    if bird_num == "5":
        ax.set_xlim(0, 38)
        ax.set_xticks(range(0, 38, 4))
    else:
        max_val = max(repeats_list) if repeats_list else 10
        ax.set_xticks(range(0, max_val + 2, 2))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=14)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

if __name__ == "__main__":
    bird_num = input("Enter bird number (1-6): ").strip()
    if bird_num not in syllables_mapping:
        print(f"Invalid bird number: {bird_num}. Please enter a number between 1 and 6.")
        sys.exit()
    yaml_path = Path(__file__).parent / ".." / "yamls" / f"bird_{bird_num}.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    repeat_distr_path = config['repeat_distr_path']
    corr_syllables = {
        "1": ['a', 'u', 'g', 'h', 'e', 'b'],
        "2": ['b', 'c', 'e'],
        "3": ['b', 'c', 'd','e'],
        "4": ['b', 'e', 'k'],
        "5": ['b', 'e', 'f'],
        "6": ['b', 'e', 'h', 'm']
    }
    unique_syllables = corr_syllables.get(bird_num)
    if unique_syllables is None:
        print("INVALID Bird Number!!!")
        sys.exit()
    df = load_and_process_data(bird_num)
    syllable_distributions = {}
    for syllable in unique_syllables:
        syllable_distributions[syllable] = get_syllable_repeats(df, syllable)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
    for idx, syllable in enumerate(unique_syllables):
        row = idx // 3
        col = idx % 3
        plot_repeat_distribution_relative(syllable_distributions[syllable], syllable, axes[row, col], bird_num)
    for idx in range(len(unique_syllables), 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
    print("\n" + "="*60)
    save_plot_choice = input("Would you like to save any specific distribution plot? (yes/no): ").strip().lower()
    if save_plot_choice in ['yes', 'y']:
        syllable_to_save = input(f"Enter the syllable to save (available: {', '.join(unique_syllables)}): ").strip().lower()
        if syllable_to_save not in unique_syllables:
            print(f"[ERROR] Syllable '{syllable_to_save}' not found. Available syllables: {unique_syllables}")
        else:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 4))
            plot_repeat_distribution_relative(syllable_distributions[syllable_to_save], syllable_to_save, ax_single, bird_num)
            from pathlib import Path
            save_dir = Path(repeat_distr_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"bird_{bird_num}_{syllable_to_save.upper()}_rpt_distr.png"
            save_path = save_dir / filename
            try:
                fig_single.savefig(save_path, dpi=300, bbox_inches='tight')
                fig_single.savefig(save_path.with_suffix('.svg'), bbox_inches='tight')
                print(f"[INFO] Figure saved successfully at:\n{save_path}")
                plt.close(fig_single)
            except Exception as e:
                print(f"[ERROR] Failed to save figure: {e}")
    else:
        print("[INFO] No plot saved.")
    import pickle
    out_dir = Path(repeat_distr_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bird_{bird_num}_repeat_distributions.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(syllable_distributions, f)
    print(f"[INFO] Repeat distributions saved to: {out_file}")
