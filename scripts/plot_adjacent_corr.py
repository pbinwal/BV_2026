"""
plot_adjacent_corr.py
=====================
Compute Spearman correlations between adjacent syllable pairs in song.

For each ordered pair of syllables (syl1, syl2), this script finds all instances
where syl1 is immediately followed by syl2 in the compressed song, extracts their
repeat counts, and computes Spearman's rank correlation.

NOTE: The significance here is without multiple comparisons correction. Corrections
are applied in pie_charts.py when compiling data from all birds together.

Usage:
    python plot_adjacent_corr.py
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, linregress
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import re
from collections import defaultdict
import os
from matplotlib.ticker import AutoMinorLocator
import yaml

from get_compressed_syntax import load_and_process_data
from get_repeat_data import syllables_mapping

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_adjacent_corr(df, unique_syllables, bird_num,
                      save_mode="default",
                      save_folder=None,
                      column="compressed_song"):

    # make a copy so we don't modify the original
    df = df.copy()

    # rename the column you want to use
    df["compressed_song"] = df[column]

    # Function to find co-occurrence frequencies for any two syllables (syl1, syl2)
    def find_repeat_cooccur_freq(sequence, syl1, syl2):
        # Initialize lists to store the numbers following syl1 and syl2
        syl1_numbers = []
        syl2_numbers = []
        
        # Extract the syllables (letters) and numbers (digits) from the sequence
        syllables = re.findall(r'[a-zA-Z]', sequence)  # Find all alphabetic characters (syllables)
        numbers = re.findall(r'\d+', sequence)         # Find all numeric characters (numbers)

        # Ensure we have the same length for syllables and numbers (the numbers should follow syllables)
        for i in range(len(syllables) - 1):  # We go till the second last syllable as the last one can't be followed by another
            # If syl2 is followed by syl1, capture the numbers after them
            if syllables[i] == syl1 and syllables[i + 1] == syl2:
                if i < len(numbers):  # Make sure we are within bounds
                    syl1_numbers.append(int(numbers[i]))  # Append the number after syl1 as integer
                if i + 1 < len(numbers):  # Make sure we are within bounds for syl2
                    syl2_numbers.append(int(numbers[i + 1]))  # Append the number after syl2 as integer

        return {syl1: syl1_numbers, syl2: syl2_numbers}

    # Main function to process the DataFrame, compute correlation, plot 
    def compute_correlation_and_plot(df, syl1, syl2, jitter_strength=0.1):
        #  Apply the function to get the co-occurrence frequencies for each song
        def apply_function_to_df(df, syl1, syl2):
            result = []
            for song in df['compressed_song']:
                cooccur_freq = find_repeat_cooccur_freq(song, syl1, syl2)
                result.append(cooccur_freq)
            return result

        # Get the co-occurrence frequencies
        repeat_cooccur_freq = apply_function_to_df(df, syl1, syl2)

        #  Add the co-occurrence frequencies to the DataFrame
        df[f'{syl1}_{syl2}_cooccur_freq'] = repeat_cooccur_freq

        # Concatenate all syl1 and syl2 lists across rows
        syl1_values = []
        syl2_values = []

        for freq_dict in df[f'{syl1}_{syl2}_cooccur_freq']:
            syl1_values += freq_dict[syl1]  # Concatenate all syl1 values
            syl2_values += freq_dict[syl2]  # Concatenate all syl2 values

        # Convert to numpy arrays
        syl1_rpt_freq = np.array(syl1_values)
        syl2_rpt_freq = np.array(syl2_values)

        # Add jitter to the values for plotting
        syl1_rpt_freq_jittered = syl1_rpt_freq + np.random.normal(0, jitter_strength, len(syl1_rpt_freq))
        syl2_rpt_freq_jittered = syl2_rpt_freq + np.random.normal(0, jitter_strength, len(syl2_rpt_freq))
    
        # Compute Spearman's correlation
        corr_coefficient, p_value = stats.spearmanr(syl1_rpt_freq, syl2_rpt_freq)

        sample_size = len(syl1_rpt_freq)

        # skip correlations with tiny sample sizes (<3)
        if sample_size < 3:               
            return np.nan, np.nan, sample_size

        # Print the correlation coefficient
        print(f"Spearmann's Correlation coefficient between {syl1} and {syl2}: {corr_coefficient} with sample size {sample_size}")

        #  Plot the relationship between syl1_rpt_freq and syl2_rpt_freq with jitter
        if(p_value < 0.05 and sample_size >= 100):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=syl1_rpt_freq_jittered, y=syl2_rpt_freq_jittered)

        # Not plotting regression line because we are using Spearman's correlation (non-parametric)

        # Set axis limits
        plt.ylim(0, 16)
        plt.xlim(0, 14)

        # Add labels and title
        plt.title(f"Scatter Plot of {syl1} repeats followed by {syl2} repeats (Correlation: {corr_coefficient:.2f})")
        plt.xlabel(f"{syl1.upper()} repeat number")
        plt.ylabel(f"{syl2.upper()} repeat number")
        
        # Adjust minor ticks
        ax = plt.gca()  # Get the current axes
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 minor ticks between major ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))  # 2 minor ticks between major ticks on y-axis

        # Display the plot
        plt.close()

        # Return the correlation coefficient and DataFrame with co-occurrence frequencies
        return corr_coefficient, p_value, sample_size


    # Generate all ordered pairs of unique syllables
    all_pairs = list(permutations(unique_syllables, 2))

    # Store both rho and p-value
    correlation_results = {}
    sample_size_dict     = {}          

    for syl1, syl2 in all_pairs:
        print(f"\nProcessing pair: {syl1}, {syl2}")
        rho, pval, n  = compute_correlation_and_plot(df, syl1, syl2)
        correlation_results[(syl1, syl2)] = {"rho": rho, "pval": pval, "n": n}
        sample_size_dict[(syl1, syl2)]    = n                          


    # Print all correlations at the end
    for pair, corr in correlation_results.items():
        print(f"{pair}: Spearman's rho = {corr['rho']:.3f}, p-value = {corr['pval']:.3f}")


    alpha = 0.05
    heatmap_data = pd.DataFrame(index=unique_syllables,
                                columns=unique_syllables,
                                dtype=float)

    # mask for grey squares only (non-significant results we DID test)
    grey_mask = pd.DataFrame(False, index=unique_syllables, columns=unique_syllables)

    for (syl1, syl2), result in correlation_results.items():
        n   = result["n"]
        rho = result["rho"]
        p   = result["pval"]
        
        # we use a threshold of n = 100 to decide if we have enough data to trust the correlation
        # this may seem overly strict but using lower thresholds would increase the number of tests we 
        # perform. When correcting for multiple comparisons this would make it harder to find significant results.

        if n < 100:                       #  "not found"  so leave cell white
            heatmap_data.loc[syl1, syl2] = np.nan          # already white in seaborn
            # keep grey_mask False so we DON'T overlay grey
        elif p < alpha and n >= 100:    #  significant  so colour by rho
            heatmap_data.loc[syl1, syl2] = rho
        else:                           #  tested but not significant
            heatmap_data.loc[syl1, syl2] = np.nan
            grey_mask.loc[syl1, syl2]    = True      # later overlay grey

    # keep the diagonal white.
    for syl in unique_syllables:
        heatmap_data.loc[syl, syl] = np.nan

    if(save_mode != "synth"): # plot only if the function is called from this script directly, not when called from synthetic data script
        # plotting 
        plt.figure(figsize=(6, 8))

        def custom_label(s):
            return s.upper()

        xticks = [custom_label(s) for s in heatmap_data.columns]
        yticks = [custom_label(s) for s in heatmap_data.index]


        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            annot_kws={"size": 14},
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=0.5,
            linecolor='white',
            square=True,
            cbar_kws={'label': 'Spearman ρ'},
            xticklabels=xticks,
            yticklabels=yticks
        )

        
        # Set colorbar (legend) text sizes 
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)          # tick label size
        cbar.set_label("Spearman ρ", fontsize=14)  # legend title size

        # overlay grey only where grey_mask is True
        for i in range(len(unique_syllables)):
            for j in range(len(unique_syllables)):
                if grey_mask.iat[i, j]:
                    plt.gca().add_patch(
                        plt.Rectangle((j, i), 1, 1,
                                    fill=True, color='darkgrey', ec='white')
                    )

             
        plt.tight_layout()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Bird {bird_num}: A to B (adjacent)\n"
                  "(grey: tested but non-significant, white: insufficient data)",
                fontsize=16)
        plt.xlabel("Syllable 2", fontsize=14)
        plt.ylabel("Syllable 1", fontsize=14)

        plt.show()

    
    ############ Save ALL adjacent correlations (significant + non-significant + insufficient data) ############

    yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    if save_mode == "default":
        output_folder_adj = os.path.join(REPO_ROOT, config.get("adjacent_dir"))
        filename = f"bird_{bird_num}_adjacent_corr_ALL.csv"
    elif save_mode == "synth":
        output_folder_adj = os.path.join(REPO_ROOT, config.get("adjacent_dir_synth"))
        filename = f"bird_{bird_num}_adjacent_corr_ALL_synth.csv"

    os.makedirs(output_folder_adj, exist_ok=True)
    output_path_all = os.path.join(output_folder_adj, filename)

    ########################## build long-format table

    rows = []

    for (syl1, syl2), result in correlation_results.items():
        rho = result["rho"]
        p   = result["pval"]
        n   = result["n"]

        if n < 100:
            status = "insufficient_data"
        elif p < 0.05:
            status = "significant"
        else:
            status = "nonsignificant"

        rows.append({
            "syl1": syl1,
            "syl2": syl2,
            "rho": rho,
            "p_value": p,
            "sample_size": n,
            "status": status
        })

    # Create the DataFrame
    all_adj_df = pd.DataFrame(rows)

    # Save CSV
    all_adj_df.to_csv(output_path_all, index=False)

    print(f"\nSaved ALL adjacent correlations (sig + non-sig + insufficient data) to:\n{output_path_all}\n")


def main():
    # Ask for bird number only when script is run directly
    available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
    bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()
    
    if bird_num not in syllables_mapping:
        print("Invalid bird number!")
        sys.exit(1)
    
    # Map bird number to unique syllables
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

    # Run adjacent correlation analysis
    run_adjacent_corr(df, unique_syllables, bird_num, save_mode="default", column="compressed_song")

if __name__ == "__main__":
    main()

    # --- Add-on: User-selected scatter plot for a syllable pair ---
    plot_pair = input("\nDo you want to plot a scatter for a specific syllable pair? (y/n): ").strip().lower()
    if plot_pair == "y":
        phrase1 = input("Enter first syllable (y-axis): ").strip().lower()
        phrase2 = input("Enter second syllable (x-axis): ").strip().lower()

        # Reload bird info and data
        bird_num = input(f"Enter bird number (1-6): ").strip()
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
        df = load_and_process_data(bird_num)

        # Use the same co-occurrence extraction as in run_adjacent_corr
        def find_repeat_cooccur_freq(sequence, syl1, syl2):
            syl1_numbers = []
            syl2_numbers = []
            syllables = re.findall(r'[a-zA-Z]', sequence)
            numbers = re.findall(r'\d+', sequence)
            for i in range(len(syllables) - 1):
                if syllables[i] == syl1 and syllables[i + 1] == syl2:
                    if i < len(numbers):
                        syl1_numbers.append(int(numbers[i]))
                    if i + 1 < len(numbers):
                        syl2_numbers.append(int(numbers[i + 1]))
            return {syl1: syl1_numbers, syl2: syl2_numbers}

        # Get co-occurrence frequencies for the selected pair
        result = []
        for song in df['compressed_song']:
            cooccur_freq = find_repeat_cooccur_freq(song, phrase1, phrase2)
            result.append(cooccur_freq)
        syl1_values = []
        syl2_values = []
        for freq_dict in result:
            syl1_values += freq_dict[phrase1]
            syl2_values += freq_dict[phrase2]
        syl1_rpt_freq = np.array(syl1_values)
        syl2_rpt_freq = np.array(syl2_values)

        # Compute Spearman's correlation
        if len(syl1_rpt_freq) >= 3 and len(syl2_rpt_freq) >= 3:
            rho, pval = stats.spearmanr(syl1_rpt_freq, syl2_rpt_freq)
            # Add jitter
            jitter_strength = 0.1
            syl1_rpt_freq_jittered = syl1_rpt_freq + np.random.normal(0, jitter_strength, len(syl1_rpt_freq))
            syl2_rpt_freq_jittered = syl2_rpt_freq + np.random.normal(0, jitter_strength, len(syl2_rpt_freq))

            plt.figure(figsize=(2.5, 2.5))
            ax = sns.scatterplot(x=syl2_rpt_freq_jittered, y=syl1_rpt_freq_jittered, s=18, edgecolor=None, alpha=0.7)
            plt.xlabel(f"{phrase2.upper()} Repeat number", fontsize=11)
            plt.ylabel(f"{phrase1.upper()} Repeat number", fontsize=11)
            plt.title(f"{phrase1.upper()} → {phrase2.upper()}", fontsize=11)
            legend_text = f"ρ = {rho:.3f}"
            leg = ax.legend([legend_text], loc='upper right', frameon=False, fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            major_tick = 4
            x_max = int(np.ceil(np.max(syl2_rpt_freq_jittered))) if len(syl2_rpt_freq_jittered) > 0 else 0
            y_max = int(np.ceil(np.max(syl1_rpt_freq_jittered))) if len(syl1_rpt_freq_jittered) > 0 else 0

            def next_major_tick(val, tick):
                return ((val // tick) + 1) * tick if val % tick != 0 else val

            x_lim = next_major_tick(x_max, major_tick)
            y_lim = next_major_tick(y_max, major_tick)

            ax.set_xlim(left=0, right=x_lim)
            ax.set_ylim(bottom=0, top=y_lim)
            ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(major_tick))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            save_dir = os.path.join(REPO_ROOT, 'figures', 'Figure 2')
            os.makedirs(save_dir, exist_ok=True)
            fname_base = 'fig2_au_panel_B'
            svg_path = os.path.join(save_dir, fname_base + '.svg')
            png_path = os.path.join(save_dir, fname_base + '.png')
            plt.savefig(svg_path, format='svg')
            plt.savefig(png_path, format='png', dpi=300)
            print(f"Saved scatter plot to:\n{svg_path}\n{png_path}")
            plt.show()
        else:
            print("Not enough data for this pair to plot.")
