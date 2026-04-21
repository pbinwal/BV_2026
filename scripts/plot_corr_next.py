"""
plot_corr_next.py
=================
Compute Spearman correlations between non-adjacent (next) syllable pairs in song.

For each ordered pair of syllables (syl1, syl2), this script finds the first
occurrence of syl2 after each block of syl1 (with no intervening syl1 block),
extracts their repeat counts, and computes Spearman's rank correlation.

NOTE: The significance here is without multiple comparisons correction. Corrections
are applied in pie_charts.py when compiling data from all birds together.

Usage:
    python plot_corr_next.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from itertools import permutations, product
import re
from collections import defaultdict
import os
from matplotlib.ticker import MaxNLocator
import yaml

from get_repeat_data import syllables_mapping
from get_compressed_syntax import load_and_process_data

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_blocks(positions):
    """ Groups consecutive occurences into rpts (blocks)
     Input: All positions of a syllable in the song """
    
    if not positions:
        return []
    
    blocks = []
    current_block = [positions[0]]
    
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1] + 1:
            current_block.append(positions[i])
        else:
            blocks.append(tuple(current_block))
            current_block = [positions[i]]
    
    blocks.append(tuple(current_block))
    return blocks


def find_syl1_syl2_pairs(songs_copy, syl1, syl2):
    ''' 
    Finds two lists:
    1. Repeat counts for syl1 (e.g., 'a').
    2. Repeat counts for the first occurrence of syl2 (e.g., 'u') after each syl1,
       ensuring no intervening syl1 block before that syl2 block.
    '''
    syl1_repeat_counts = []
    syl2_repeat_counts = []

    songs = songs_copy["song"]

    for song in songs:
        syl1_positions = [i for i, syllable in enumerate(song) if syllable == syl1]
        syl2_positions = [i for i, syllable in enumerate(song) if syllable == syl2]

        syl1_blocks = get_blocks(syl1_positions)
        syl2_blocks = get_blocks(syl2_positions)

        if syl1 == syl2:
            # For self-pairs, pair each block with the next block in the same song
            for k in range(len(syl1_blocks) - 1):
                syl1_repeat_counts.append(len(syl1_blocks[k]))
                syl2_repeat_counts.append(len(syl1_blocks[k + 1]))
        else:
            for k, a_block in enumerate(syl1_blocks):
                last_syl1 = a_block[-1]
                syl1_repeat_count = len(a_block)

                # Determine when the next syl1 block starts (if any)
                next_a_start = syl1_blocks[k + 1][0] if (k + 1) < len(syl1_blocks) else float('inf')

                # Find the first syl2 block that occurs after the current syl1 block
                # and before the next syl1 block starts (no intervening syl1 block)
                for b_block in syl2_blocks:
                    b_start = b_block[0]
                    if b_start > last_syl1 and b_start < next_a_start:
                        first_syl2 = b_start
                        syl2_repeat_count = len(b_block)

                        syl1_repeat_counts.append(syl1_repeat_count)
                        syl2_repeat_counts.append(syl2_repeat_count)

                        break  # Only consider the first valid syl2 after syl1

    return syl1_repeat_counts, syl2_repeat_counts


def add_jitter(data, jitter_strength=0.1):
    return data + np.random.uniform(-jitter_strength, jitter_strength, size=len(data))


def run_next_corr(df, unique_syllables, bird_num, save_mode="default", save_folder=None, column="song"):
    """
    Compute correlations between syl1 and the first next occurrence of syl2.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing song data
    unique_syllables : list
        List of unique syllables to analyze
    bird_num : str
        Bird number (1-6)
    save_mode : str
        Either "default" for real data or "synth" for synthetic data
    save_folder : str, optional
        Custom save folder (not used currently, for future expansion)
    column : str
        Column name containing song data (default: "song")
    """
    
    # Make a copy so we don't modify the original
    df = df.copy()
    
    # Rename the column you want to use
    df["song"] = df[column]
    
    # Define minimum sample size threshold
    MIN_SAMPLE_SIZE = 100
    
    valid_pairs = [(s1, s2) for s1, s2 in product(unique_syllables, repeat=2)] # to include self-pairs

    n = len(unique_syllables)
    corr_matrix = np.full((n, n), np.nan)
    p_val_matrix = np.full((n, n), np.nan)
    size_matrix = np.full((n, n), np.nan)

    for idx, (syl1, syl2) in enumerate(valid_pairs):

        syl1_counts, syl2_counts = find_syl1_syl2_pairs(df, syl1, syl2)

        # # Debug: Print info for self-pairs (syl1 == syl2)
        # if syl1 == syl2:
        #     print(f"[DEBUG-SELF] {syl1.upper()} self-pair: sample size = {len(syl1_counts)}")
        #     if len(syl1_counts) > 0:
        #         print(f"[DEBUG-SELF] Example pairs: {list(zip(syl1_counts, syl2_counts))[:5]}")

        if len(syl1_counts) < MIN_SAMPLE_SIZE or len(syl1_counts) != len(syl2_counts):
            # Print which pair is being skipped and why
            if len(syl1_counts) != len(syl2_counts):
                reason = "list length mismatch"
            else:
                reason = f"n={len(syl1_counts)} < {MIN_SAMPLE_SIZE}"
            print(f"Skipping pair {syl1.upper()}->{syl2.upper()}: {reason}")
            
            i = unique_syllables.index(syl1)
            j = unique_syllables.index(syl2)
            corr_matrix[i, j] = np.nan
            p_val_matrix[i, j] = np.nan
            size_matrix[i, j] = len(syl1_counts)
            continue

        i = unique_syllables.index(syl1)
        j = unique_syllables.index(syl2)

        r_value, p_value = spearmanr(syl1_counts, syl2_counts)
        corr_matrix[i, j] = r_value
        p_val_matrix[i, j] = p_value
        size_matrix[i, j] = len(syl1_counts)

    # Convert to DataFrame with syllable names as labels
    corr_df = pd.DataFrame(corr_matrix, index=unique_syllables, columns=unique_syllables)
    p_val_df = pd.DataFrame(p_val_matrix, index=unique_syllables, columns=unique_syllables)
    size_df = pd.DataFrame(size_matrix, index=unique_syllables, columns=unique_syllables)

    print("\ncorr_df\n", corr_df)
    print("\np_val_df\n", p_val_df)

    # Heatmap plotting code removed as requested. All calculations and saving remain unchanged.

    ############ Save ALL correlations (significant + non-significant + insufficient data) ############

    yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    if save_mode == "default":
        output_folder_next = os.path.join(REPO_ROOT, config.get("next_dir"))
        bird_label = f"bird_{bird_num}"
        filename = f"{bird_label}_next_corr_ALL.csv"
    elif save_mode == "synth":
        output_folder_next = os.path.join(REPO_ROOT, config.get("next_dir_synth"))
        bird_label = f"bird_{bird_num}"
        filename = f"{bird_label}_next_corr_ALL_synth.csv"
    
    os.makedirs(output_folder_next, exist_ok=True)
    output_path_all = os.path.join(output_folder_next, filename)

    rows = []

    for s1 in unique_syllables:
        for s2 in unique_syllables:
            corr_val = corr_df.loc[s1, s2] if s1 in corr_df.index and s2 in corr_df.columns else np.nan
            p_val    = p_val_df.loc[s1, s2] if s1 in p_val_df.index and s2 in p_val_df.columns else np.nan
            n_val    = size_df.loc[s1, s2] if s1 in size_df.index and s2 in size_df.columns else np.nan

            # Determine status
            if pd.isna(n_val) or n_val < MIN_SAMPLE_SIZE:
                status = "insufficient_data"
            elif not pd.isna(p_val) and p_val < 0.05:
                status = "significant"
            else:
                status = "nonsignificant"

            rows.append({
                "syl1": s1,
                "syl2": s2,
                "correlation": corr_val,
                "p_value": p_val,
                "sample_size": n_val,
                "status": status
            })

    all_corr_df = pd.DataFrame(rows)
    all_corr_df.to_csv(output_path_all, index=False)
    print(f"\nSaved ALL NEXT correlations (sig + non-sig + insufficient data) to:\n{output_path_all}\n")


def main():
    # Ask for bird number only when script is run directly
    available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
    bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()
    
    if bird_num not in syllables_mapping:
        print("Invalid bird number!")
        sys.exit(1)
    
    # Map bird number to unique syllables for correlation analysis
    corr_syllables = {
        "1": ['a', 'u', 'g', 'h', 'e', 'b'],
        "2": ['b', 'c', 'e'],
        "3": ['b', 'c', 'd', 'e'],
        "4": ['b', 'e', 'k'],
        "5": ['b', 'e', 'f'],
        "6": ['b', 'e', 'h', 'm']
    }
    
    unique_syllables = corr_syllables.get(bird_num)
    if unique_syllables is None:
        print("Invalid bird number!")
        sys.exit(1)
    
    # Load the real dataset
    df = load_and_process_data(bird_num)

    # Run next correlation analysis
    run_next_corr(df, unique_syllables, bird_num, save_mode="default", column="song")

if __name__ == "__main__":
    main()
