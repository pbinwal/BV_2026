"""
z_test_corr.py
==============
Z-test for correlations using synthetic data generation.

This script validates observed correlations by generating 100 synthetic song
datasets. Each synthetic dataset preserves the empirical repeat-number
distributions and the song sequences. This means that keeping the sequence of phrases
intact, we sample the repeat number of phrases from their characteristic repeat
distributions to test whether observed correlations are simply byproducts
of the constraints imposed by repeat distrbutions of phrases.

Correlations computed on these synthetic datasets form a null
distribution against which the real correlation is tested (one-sided z-test).

Usage:
    python z_test_corr.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from itertools import permutations
import re
from collections import defaultdict
import os
from scipy.stats import norm
import yaml
from pathlib import Path

# Import module functions
from get_compressed_syntax import load_and_process_data
from fig_2_b_plot_adjacent_corr import run_adjacent_corr
from plot_corr_next import run_next_corr
from get_repeat_data import syllables_mapping

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bird number input
available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()

# Validate bird_num
if bird_num not in syllables_mapping:
    print("Invalid bird number!")
    sys.exit(1)

# Load YAML configuration
yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# Define syllables based on bird
if bird_num == "1":
    rpt_syllables = ['a', 'u', 'g', 'h', 'e', 'b']
elif bird_num == "2":
    rpt_syllables = ['b', 'c', 'e']
elif bird_num == "3":
    rpt_syllables = ['b', 'c', 'd', 'e']
elif bird_num == "4":
    rpt_syllables = ['b', 'e', 'k']
elif bird_num == "5":
    rpt_syllables = ['b', 'e', 'f']
elif bird_num == "6":
    rpt_syllables = ['b', 'e', 'h', 'm']

# Ask user which analysis to run
analysis_type = input("Enter 1 for Adjacent correlation or 2 for Next correlation: ").strip()

if analysis_type not in ["1", "2"]:
    print("INVALID choice! Must be 1 or 2.")
    sys.exit(1)

# Load processed data (with compressed songs)
df = load_and_process_data(bird_num)

# Treat any 'y' as 'd' for bird 3 because I renamed that syllable in all other scripts
if bird_num == "3":
    df["compressed_song"] = df["compressed_song"].str.replace("y", "d")

df = df.reset_index(drop=True)  # fix missing indices

# Auto-detect unique syllables from compressed songs
all_syllables = df["compressed_song"].str.cat()
unique_syllables = sorted(set(re.findall(r"[a-zA-Z]", all_syllables)))

print("Detected syllables:", unique_syllables)

##############
def get_rpt_num_prob(repeat_num_list):
    """Function to get the probabilities of each number in a list"""
    total_count = len(repeat_num_list)
    num_counts = {}
    for num in repeat_num_list:
        num_counts[num] = num_counts.get(num, 0) + 1
    num_probs = {num: count / total_count for num, count in num_counts.items()}
    sorted_num_probs = dict(sorted(num_probs.items()))
    return sorted_num_probs

##############
def get_syllable_repeats(df, syllable):
    """
    This function finds and gets the repeat numbers for all repeats of a syllable in the data frame

    -Inputs
        df: data frame containing a compressed_song column
        syllable: the syllable whose repeat number you are interested in

    -Returns
        repeats_list: a list containing all repeat numbers of a syllable in the data frame
    """
    repeats_list = []
    for song in df['compressed_song']:
        matches = re.findall(rf'{syllable}(\d+)', song)
        repeats = [int(match) for match in matches]
        repeats_list.extend(repeats)
    return repeats_list

# Dynamically get repeat distributions for all unique_syllables
rpt_distr_for_all_syl = []

for syl in unique_syllables:
    repeats = get_syllable_repeats(df, syl)
    rpt_distr = get_rpt_num_prob(repeats)
    rpt_distr_for_all_syl.append(rpt_distr)

# Corresponding syllable names
syllable_names = unique_syllables

# Dictionary to hold the results
rpt_prob_dict = {}

for rpt_distr, syllable_name in zip(rpt_distr_for_all_syl, syllable_names):
    rpt_prob = get_rpt_num_prob(rpt_distr)
    rpt_prob_dict[syllable_name] = rpt_prob

choice = input("Enter y to re-calculate correlations in simulated data and re-save/ overwrite possibly existing csv's OR enter n to use existing data: ").strip()
if choice == "y":

    # Duplicate the original DataFrame
    df_copy = df.copy()
    df_copy['synth_song'] = [''] * len(df_copy)

    np.random.seed(42)

    # Function to decompress compressed songs
    def decompress_song(compressed_song):
        """
        Converts compressed song format to uncompressed list format.
        Example: "a3b2u4" -> ['a','a','a','b','b','u','u','u','u']
        """
        uncompressed = []
        matches = re.findall(r'([a-zA-Z])(\d+)', compressed_song)
        for syllable, count in matches:
            uncompressed.extend([syllable] * int(count))
        return uncompressed

    ##############
    # Run correlation analysis on the synthetic song
    n = 100  # number of synthetic simulations

    for iteration in range(1, n+1):
        print(f"\n=== Synthetic song iteration {iteration} ===")

        # Generate synthetic songs
        df_copy['synth_song'] = [''] * len(df_copy)
        np.random.seed(42 + iteration)

        for idx, song in enumerate(df_copy["song"]):
            for i, syl in enumerate(song):
                if syl != "I":
                    syl = syl.lower()
                    values = list(rpt_prob_dict[syl].keys())
                    probabilities = list(rpt_prob_dict[syl].values())
                    sample = np.random.choice(values, size=1, p=probabilities)
                    df_copy.at[idx, 'synth_song'] += syl + str(sample[0])
                else:
                    df_copy.at[idx, 'synth_song'] += syl + str(1)

        # Run correlation analysis based on user choice
        if analysis_type == "1":
            run_adjacent_corr(
                df_copy,
                rpt_syllables,
                bird_num,
                save_mode="synth",
                column="synth_song"
            )
            # Move file to bird_num subdirectory with iteration number in filename
            synth_base = os.path.join(REPO_ROOT, config.get('adjacent_dir_synth'))
            source_file = os.path.join(synth_base, f"bird_{bird_num}_adjacent_corr_ALL_synth.csv")
            target_dir = os.path.join(synth_base, f"bird_{bird_num}")
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, f"bird_{bird_num}_{iteration}_adjacent_corr_ALL_synth.csv")
            if os.path.exists(source_file):
                import shutil
                shutil.move(source_file, target_file)
                
        elif analysis_type == "2":
            df_copy['synth_song_uncompressed'] = df_copy['synth_song'].apply(decompress_song)
            run_next_corr(
                df_copy,
                rpt_syllables,
                bird_num,
                save_mode="synth",
                column="synth_song_uncompressed"
            )
            # Move file to bird_num subdirectory with iteration number in filename
            synth_base = os.path.join(REPO_ROOT, config.get('next_dir_synth'))
            source_file = os.path.join(synth_base, f"bird_{bird_num}_next_corr_ALL_synth.csv")
            target_dir = os.path.join(synth_base, f"bird_{bird_num}")
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, f"bird_{bird_num}_{iteration}_next_corr_ALL_synth.csv")
            if os.path.exists(source_file):
                import shutil
                shutil.move(source_file, target_file)

    print(f"Completed 100 synthetic song iterations and saved {'adjacent' if analysis_type == '1' else 'next'} correlation CSVs.")

elif choice == "n":
    pass
else:
    print("Invalid Choice! Aborting script")
    sys.exit(1)

######### Plot distributions of correlations from simulated data

# Get paths from YAML (resolved relative to repo root)
if analysis_type == "1":
    synth_folder = os.path.join(REPO_ROOT, config.get('adjacent_dir_synth'))
    real_data_dir = os.path.join(REPO_ROOT, config.get('adjacent_dir'))
    if not synth_folder or not real_data_dir:
        print("ERROR: Missing 'adjacent_dir_synth' or 'adjacent_dir' in YAML")
        sys.exit(1)
    real_data_path = os.path.join(real_data_dir, f"bird_{bird_num}_adjacent_corr_ALL.csv")
    filename_suffix = "_adjacent_corr_ALL_synth.csv"
    analysis_name = "Adjacent"
    corr_col = 'rho'
elif analysis_type == "2":
    synth_folder = os.path.join(REPO_ROOT, config.get('next_dir_synth'))
    real_data_dir = os.path.join(REPO_ROOT, config.get('next_dir'))
    if not synth_folder or not real_data_dir:
        print("ERROR: Missing 'next_dir_synth' or 'next_dir' in YAML")
        sys.exit(1)
    real_data_path = os.path.join(real_data_dir, f"bird_{bird_num}_next_corr_ALL.csv")
    filename_suffix = "_next_corr_ALL_synth.csv"
    analysis_name = "Next"
    corr_col = 'correlation'

n = 100  # number of synthetic iterations

# Load real data correlations
df_real = pd.read_csv(real_data_path)

# Generate all ordered pairs (permutations) of syllables
all_pairs = list(permutations(rpt_syllables, 2))

for syllable_pair in all_pairs:
    correlation_values = []

    # Collect synthetic correlations across all iterations
    for i in range(1, n+1):
        csv_path = os.path.join(synth_folder, f"bird_{bird_num}", f"bird_{bird_num}_{i}{filename_suffix}")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            df_pair = df[(df['syl1'] == syllable_pair[0]) &
                         (df['syl2'] == syllable_pair[1]) &
                         (df['status'] != 'insufficient_data')]
            if not df_pair.empty:
                correlation_values.extend(df_pair[corr_col].tolist())
        else:
            print(f"File not found: {csv_path}")

    # Get real correlation for this pair
    df_real_pair = df_real[(df_real['syl1'] == syllable_pair[0]) &
                           (df_real['syl2'] == syllable_pair[1]) &
                           (df_real['status'] != 'insufficient_data')]

    if correlation_values:
        # Check if real data correlation exists and is significant
        if not df_real_pair.empty and df_real_pair['status'].values[0] == "significant":
            plt.figure(figsize=(8,6))
            sns.histplot(correlation_values, bins=20, kde=True, color='skyblue')

            # Add vertical line for real data correlation
            real_rho = df_real_pair[corr_col].values[0]
            plt.axvline(real_rho, color='red', linestyle='--', linewidth=2, label='Observed correlation')

            # Perform one-sided z-test
            synth_mean = np.mean(correlation_values)
            synth_std = np.std(correlation_values, ddof=1)
            if synth_std == 0:
                p_value = np.nan
            else:
                z_score = (real_rho - synth_mean) / synth_std
                if real_rho > 0:
                    p_value = 1 - norm.cdf(z_score)  # right-tailed
                else:
                    p_value = norm.cdf(z_score)      # left-tailed

            # Annotate p-value
            plt.text(0.05, 0.90, f"One-sided p-value: {p_value:.4f}", transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            plt.legend(loc='upper right')

            plt.title(f"Distribution of Spearman Correlations for {syllable_pair[0]} → {syllable_pair[1]}\n({analysis_name} Analysis)")
            plt.xlabel("Spearman ρ")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
            plt.show()
        else:
            print(f"Skipped {syllable_pair[0]} → {syllable_pair[1]}: Observed correlation not significant")
    else:
        print(f"Skipped {syllable_pair[0]} → {syllable_pair[1]}: no sufficient synthetic data in any iteration")
