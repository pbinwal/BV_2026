"""
Fig 1: Prceeding and Upcoming context plots in panels: C, D, E, F
Run for Bird 1: U, which syl repeats= 1 and Bird 1: E, which syl repeats = 2 to reproduce Fig 1 D and F
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
import itertools
import yaml
import os

# Import the new module function
from get_compressed_syntax import load_and_process_data
from get_repeat_data import syllables_mapping


# Bird number input (or could be command-line argument)
bird_id = input("Enter bird ID (1-6): ").strip()
print(f"[DEBUG] Bird number entered: {bird_id}")

# Validate bird number and get unique syllables
if bird_id not in syllables_mapping:
    print("[DEBUG] INVALID Bird Number!!! Exiting.")
    sys.exit()

unique_syllables = syllables_mapping[bird_id]
print(f"[DEBUG] Unique syllables for bird {bird_id}: {unique_syllables}")

# Load processed data (with compressed songs)
df = load_and_process_data(bird_id)
print(f"[DEBUG] Loaded dataframe with {len(df)} rows")

print("[DEBUG] First few compressed songs:", df['compressed_song'].head().tolist())


# Load YAML configuration
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yaml_path = os.path.join(repo_root, "yamls", f"bird_{bird_id}.yaml")

if not os.path.isfile(yaml_path):
    print(f"ERROR: YAML file not found: {yaml_path}")
    sys.exit(1)

with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

transition_dir = config.get("transition_dir")
if not transition_dir:
    print("ERROR: 'transition_dir' not found in YAML file.")
    sys.exit(1)

repeat_distr_path = config.get("repeat_distr_path")
if not repeat_distr_path:
    print("ERROR: 'repeat_distr_path' not found in YAML file.")
    sys.exit(1)

probs_file = os.path.join(transition_dir, f"bird_{bird_id}_transition_probs.npy")
syll_file = os.path.join(transition_dir, f"bird_{bird_id}_unique_syllables.npy")

# Check if files exist
if not os.path.isfile(probs_file):
    print(f"ERROR: Transition probabilities file not found: {probs_file}")
    print("Please run plot_trans_diag.py first to generate the transition matrices.")
    sys.exit(1)

if not os.path.isfile(syll_file):
    print(f"ERROR: Unique syllables file not found: {syll_file}")
    print("Please run plot_trans_diag.py first to generate the transition matrices.")
    sys.exit(1)

print(f"[DEBUG] Loading saved transition matrices from {transition_dir}")
transition_probs = np.load(probs_file)
unique_syllables_all = np.load(syll_file).tolist()
print(f"[DEBUG] Loaded transition probabilities shape: {transition_probs.shape}")
print(f"[DEBUG] Loaded all unique syllables from file: {unique_syllables_all}")


# Use mapping for target syllables for context analysis
target_syllables = syllables_mapping[bird_id]
print(f"[DEBUG] Target syllables for analysis: {target_syllables}")
print(f"[DEBUG] Loaded transition probabilities for bird {bird_id} from:\n{probs_file}")

# Use all syllables for transition matrix indexing, but only analyze target syllables
unique_syllables = unique_syllables_all

# Function to get the repeat distribution for syllable pairs where the second syllable repeats
def get_rpts_cntxt_aft(df, syl1, syl2):
    print(f"[DEBUG] Getting repeats where '{syl2}' follows '{syl1}'")
    repeats_list = []
    
    for i, song in enumerate(df['compressed_song']):
        pattern = rf'({syl1}\d+){syl2}(\d+)'  
        matches = re.findall(pattern, song, flags=re.IGNORECASE)
        repeats = [int(match[1]) for match in matches]  
        repeats_list.extend(repeats)
        if matches:
            print(f"[DEBUG] Song {i}: found {len(matches)} matches for pattern '{pattern}'")
    print(f"[DEBUG] Total repeats found (after '{syl1}' and before '{syl2}'): {len(repeats_list)}")
    return repeats_list

def get_rpts_cntxt_bef(df, syl1, syl2):
    print(f"[DEBUG] Getting repeats where '{syl1}' precedes '{syl2}'")
    repeats_list = []
    
    for i, song in enumerate(df['compressed_song']):
        pattern = rf'{syl1}(\d+){syl2}(\d+)'  
        matches = re.findall(pattern, song, flags=re.IGNORECASE)
        repeats = [int(match[0]) for match in matches]  
        repeats_list.extend(repeats)
        if matches:
            print(f"[DEBUG] Song {i}: found {len(matches)} matches for pattern '{pattern}'")
    print(f"[DEBUG] Total repeats found (before '{syl2}' after '{syl1}'): {len(repeats_list)}")
    return repeats_list


# List of syllables
syllables = unique_syllables

# Generate all possible pairs of syllables (permutations of length 2)
syllable_pairs = itertools.permutations(syllables, 2)

# Create dictionaries to store results for each syllable pair 
repeats_by_pair_aft = {}
repeats_by_pair_bef = {}

print("[DEBUG] Computing repeats for all syllable pairs...")
for syl1, syl2 in syllable_pairs:
    key = f'{syl1}{syl2}'
    repeats_aft = get_rpts_cntxt_aft(df, syl1, syl2)
    repeats_bef = get_rpts_cntxt_bef(df, syl1, syl2)
    repeats_by_pair_aft[key] = repeats_aft
    repeats_by_pair_bef[key] = repeats_bef
    # print(f"[DEBUG] Pair {key}: repeats after count = {len(repeats_aft)}, repeats before count = {len(repeats_bef)}")

print("[DEBUG] Finished computing repeat distributions for all pairs.")


def plot_repeat_distribution_relative(repeats_list, syllable, ax, which_syl_rpts, color="#d3d3d3", pct_text="", max_repeat=None, y_max=1.0):
    print(f"[DEBUG] Plotting repeat distribution for pair {syllable} with which_syl_rpts={which_syl_rpts}")
    if not repeats_list:
        print(f"[DEBUG] Empty repeats list for {syllable}. Skipping plot.")
        return
    
    # Determine the bin range based on max_repeat if provided
    if max_repeat is not None:
        bin_max = max_repeat + 1
    elif bird_id == "5":
        bin_max = 38
    else:
        bin_max = 23
    
    bins = np.arange(0 - 0.5, bin_max + 1.5, 1)
    ax.hist(repeats_list, bins=bins, edgecolor='black', alpha=0.7, density=True, color=color)


    def map_syllable_for_title(s):  # keep as it is
        return s

    syl1 = map_syllable_for_title(syllable[0].upper())
    syl2 = map_syllable_for_title(syllable[1].upper())

    # Determine which syllable repeats
    if which_syl_rpts == 1:
        # First syllable repeats: title is the repeating syllable
        title_text = syl1
        subtitle_text = f"$\\mathbf{{{syl1}}}$ → {syl2}"
    else:
        # Second syllable repeats
        title_text = syl2
        subtitle_text = f"{syl1} → $\\mathbf{{{syl2}}}$"

    ax.set_title(f"{subtitle_text}{pct_text}", fontsize=28)
    ax.set_xlabel('Repeat Number', fontsize=32)
    ax.set_ylabel('Relative Frequency', fontsize=32)

    # Set x-axis limits and ticks based on max_repeat if provided
    if max_repeat is not None:
        ax.set_xlim(0, max_repeat + 1)
        major_ticks = range(0, max_repeat + 2, 4)
        minor_ticks = range(0, max_repeat + 2, 2)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
    elif bird_id == "5": # this bird has long repeats
        ax.set_xlim(0, 38)
        ax.set_xticks(range(0, 40, 4))       # major ticks every 4
        ax.set_xticks(range(0, 40, 2), minor=True)  # minor ticks every 2
    else:
        ax.set_xticks(range(0, 24, 4))       # major ticks every 4
        ax.set_xticks(range(0, 24, 2), minor=True)  # minor ticks every 2

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='x', which='minor', length=4, width=1)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.4))
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.2), minor=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"[DEBUG] Finished plotting {title_text}")


def plot_all_repeat_distributions_separate(repeats_by_pair, transition_probs, unique_syllables, which_syl_rpts):
    print(f"[DEBUG] Plotting all repeat distributions grouped by syllable #{which_syl_rpts}")
    grouped_by_syllable_end = defaultdict(list)
    
    # Create index with consistent case handling
    syllable_index = {}
    for i, syl in enumerate(unique_syllables):
        # Convert all to uppercase for consistent comparison
        syllable_index[syl.upper()] = i

    print(f"[DEBUG] Syllable index mapping: {syllable_index}")

    for pair, repeats_list in repeats_by_pair.items():
        syl1, syl2 = pair
        
        # Convert both syllables to uppercase for consistent comparison
        syl1_upper = syl1.upper()
        syl2_upper = syl2.upper()
        
        # Handle any character substitutions consistently
        if bird_id == "1":
            syl1_upper = syl1_upper.replace('I', 'Q')
            syl2_upper = syl2_upper.replace('I', 'Q')

        if syl1_upper not in syllable_index or syl2_upper not in syllable_index:
            print(f"[DEBUG] Skipping pair {pair}: {syl1_upper} or {syl2_upper} not in index")
            continue
        
        idx_1 = syllable_index[syl1_upper]
        idx_2 = syllable_index[syl2_upper]

        # Check if transition probability is greater than 5%
        if transition_probs[idx_1, idx_2] > 0.05:  # <-- ADDED 5% THRESHOLD
            # Only include if the repeating syllable is in target_syllables
            if which_syl_rpts == 1 and syl1.lower() in target_syllables:
                grouped_by_syllable_end[syl1_upper].append((pair, repeats_list))
            elif which_syl_rpts == 2 and syl2.lower() in target_syllables:
                grouped_by_syllable_end[syl2_upper].append((pair, repeats_list))
        else:
            print(f"[DEBUG] Skipping pair {pair}: transition probability {transition_probs[idx_1, idx_2]:.3f} < 5%")

    print(f"[DEBUG] Number of syllable groups to plot: {len(grouped_by_syllable_end)}")

    # Dictionary to store figures for each syllable
    syllable_figures = {}

    for syl, syllable_pairs in grouped_by_syllable_end.items():
        print(f"[DEBUG] Plotting syllable '{syl}' with {len(syllable_pairs)} pairs")
        
        # Calculate the maximum repeat number across all contexts for this syllable
        max_repeat = 0
        for pair, repeats_list in syllable_pairs:
            if repeats_list:
                max_repeat = max(max_repeat, max(repeats_list))
        
        print(f"[DEBUG] Max repeat number for syllable '{syl}': {max_repeat}")
        
        # Calculate percentages for this syllable group
        percentages = calculate_percentages_for_syllable_group(repeats_by_pair, syl.lower(), which_syl_rpts)
        
        num_rows = 1
        max_cols = len(syllable_pairs)

        fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 6, num_rows * 4), sharey=True)
        if max_cols == 1:
            axes = [axes]

        _syl_key = syl.lower()
        if bird_id == "1" and _syl_key == "e" and which_syl_rpts == 2:
            _color = "#e69f00"
            _y_max = 0.8
        elif bird_id == "1" and _syl_key == "u" and which_syl_rpts == 1:
            _color = "#4682b4"
            _y_max = 1.0
        else:
            _color = "#d3d3d3"
            _y_max = 1.0

        for ax_counter, (pair, repeats_list) in enumerate(syllable_pairs):
            pct_text = ""
            if pair in percentages:
                pct_text = f" ({percentages[pair]:.1f}%)"
            
            plot_repeat_distribution_relative(
                repeats_list, 
                pair, 
                axes[ax_counter], 
                which_syl_rpts,
                color=_color,
                pct_text=pct_text,
                max_repeat=max_repeat,
                y_max=_y_max,
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Store the figure for this syllable
        syllable_figures[syl.lower()] = fig
        
        plt.show()
    
    return syllable_figures, grouped_by_syllable_end

def calculate_percentages_for_syllable_group(repeats_by_pair, syllable, which_syl_rpts):
    """
    Calculate % of repeats for each pair that belongs to the syllable group
    defined by which_syl_rpts and syllable.
    """
    # Convert target syllable to uppercase for consistent comparison
    target_syllable = syllable.upper()
    if bird_id == "1":
        target_syllable = target_syllable.replace('I', 'Q')
    
    relevant_pairs = {}
    for pair, repeats in repeats_by_pair.items():
        syl1, syl2 = pair
        syl1_upper = syl1.upper()
        syl2_upper = syl2.upper()
        
        if bird_id == "1":
            syl1_upper = syl1_upper.replace('I', 'Q')
            syl2_upper = syl2_upper.replace('I', 'Q')
        
        if which_syl_rpts == 1 and syl1_upper == target_syllable:
            relevant_pairs[pair] = repeats
        elif which_syl_rpts == 2 and syl2_upper == target_syllable:
            relevant_pairs[pair] = repeats

    total_count = sum(len(repeats) for repeats in relevant_pairs.values())
    if total_count == 0:
        return {}

    percentages = {pair: (len(repeats) / total_count) * 100 for pair, repeats in relevant_pairs.items()}
    
    print(f"[DEBUG] Syllable '{syllable}' contexts:")
    total_pct = 0
    for pair, pct in percentages.items():
        print(f"  {pair}: {pct:.1f}% ({len(relevant_pairs[pair])} occurrences)")
        total_pct += pct
    print(f"  Total: {total_pct:.1f}%")
    
    return percentages

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

def plot_all_repeat_distributions_together(repeats_by_pair, transition_probs, unique_syllables, which_syl_rpts):
    print(f"[DEBUG] Plotting combined repeat distributions grouped by syllable #{which_syl_rpts}")
    grouped_by_syllable_end = defaultdict(list)
    syllable_index = {syl.upper(): i for i, syl in enumerate(unique_syllables)}

    # Group valid pairs by syllable end (1 or 2)
    for pair, repeats_list in repeats_by_pair.items():
        syl1, syl2 = pair
        if(bird_id == "1"):
            syl1 = syl1.upper().replace('I', 'Q')
            syl2 = syl2.upper().replace('I', 'Q')

        if syl1 not in syllable_index or syl2 not in syllable_index:
            continue

        idx_1 = syllable_index[syl1]
        idx_2 = syllable_index[syl2]

        # ADDED: Check if transition probability is greater than 5%
        if transition_probs[idx_1, idx_2] > 0.05 and repeats_list:  # Only valid pairs with data and >5% probability
            if which_syl_rpts == 1:
                grouped_by_syllable_end[syl1].append((pair, repeats_list))
            elif which_syl_rpts == 2:
                grouped_by_syllable_end[syl2].append((pair, repeats_list))
        else:
            print(f"[DEBUG] Skipping pair {pair}: transition probability {transition_probs[idx_1, idx_2]:.3f} < 5% or no data")

    # Combine all valid pairs from all groups into one list
    all_pairs = []
    for syllable_pairs in grouped_by_syllable_end.values():
        all_pairs.extend(syllable_pairs)

    if not all_pairs:
        print("[DEBUG] No valid pairs to plot in combined figure.")
        return

    # Create color map for syllables that group the pairs
    unique_group_sylls = list(grouped_by_syllable_end.keys())
    cmap = cm.get_cmap('tab20', len(unique_group_sylls))  # 20 distinct colors max

    syllable_to_color = {syl: cmap(i) for i, syl in enumerate(unique_group_sylls)}

    print(f"[DEBUG] Plotting combined figure with {len(all_pairs)} pairs")
    num_pairs = len(all_pairs)
    ncols = min(6, num_pairs)  # max 6 columns
    nrows = (num_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2), sharey=True)
    axes = axes.flatten() if num_pairs > 1 else [axes]

    for ax in axes[num_pairs:]:
        ax.axis('off')

    # Plot each pair's repeat distribution on its respective axis
    for ax_counter, (pair, repeats_list) in enumerate(all_pairs):
        # Determine which syllable controls the color
        color_syl = pair[0].upper().replace('I', 'Q') if which_syl_rpts == 1 else pair[1].upper().replace('I', 'Q')
        color = syllable_to_color.get(color_syl, 'black')  # fallback black if not found

        # Assuming plot_repeat_distribution_relative can accept a color argument
        plot_repeat_distribution_relative(repeats_list, pair, axes[ax_counter], which_syl_rpts, color=color)
        axes[ax_counter].set_xlim(0, 13)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Ask user for which syllable repeats: 1 or 2
while True:
    which_syl_rpts_input = input("Enter which syllable repeats (1 or 2): ").strip()
    if which_syl_rpts_input in ['1', '2']:
        which_syl_rpts = int(which_syl_rpts_input)
        break
    else:
        print("Invalid input. Please enter 1 or 2.")

# Select repeats_by_pair dictionary based on user input
if which_syl_rpts == 1:
    repeats_by_pair = repeats_by_pair_bef
else:
    repeats_by_pair = repeats_by_pair_aft


# plot_all_repeat_distributions_together(repeats_by_pair, transition_probs, unique_syllables, which_syl_rpts)
figures_dict, grouped_by_syllable_end = plot_all_repeat_distributions_separate(repeats_by_pair, transition_probs, unique_syllables, which_syl_rpts)


############# comparing the distributions in different contexts using k-sample anderson darling test
from scipy.stats import anderson_ksamp, ks_2samp

from scipy.stats import anderson_ksamp

print("\nPerforming single k-sample Anderson-Darling test per syllable across all contexts...")

# Dictionary to store AD test results
ad_test_results = {}

for syl in target_syllables:
    # Collect all repeats for this syllable across contexts
    if which_syl_rpts == 1:
        pairs_for_syl = {pair: repeats for pair, repeats in repeats_by_pair.items() if pair[0].lower() == syl.lower() and repeats}
    else:
        pairs_for_syl = {pair: repeats for pair, repeats in repeats_by_pair.items() if pair[1].lower() == syl.lower() and repeats}

    if len(pairs_for_syl) == 0:
        print(f"Syllable '{syl}': no contexts with data, skipping")
        continue

    # Extract repeat lists for all contexts
    samples = [repeats for repeats in pairs_for_syl.values()]

    try:
        # Perform single k-sample A-D test across all contexts
        ad_result = anderson_ksamp(samples)
        ad_test_results[syl.upper()] = {
            "statistic": ad_result.statistic,
            "p_value": ad_result.pvalue
        }
        print(f"Syllable '{syl}': AD statistic = {ad_result.statistic:.3f}, p-value = {ad_result.pvalue:.3g}")
    except Exception as e:
        print(f"Syllable '{syl}': Error performing AD test: {e}")




# Create syllable index for transition matrix lookup
syllable_index = {syl.upper(): i for i, syl in enumerate(unique_syllables)}

# Step 1: Filter valid pairs (transition probability > 5% and non-empty repeats)
# Only include pairs where the repeating syllable is in target_syllables
valid_pairs = {}
for pair, repeats_list in repeats_by_pair.items():
    syl1, syl2 = pair
    idx_1 = syllable_index[syl1.upper()]
    idx_2 = syllable_index[syl2.upper()]
    
    # Check if repeating syllable is in target list
    is_target = False
    if which_syl_rpts == 1 and syl1.lower() in target_syllables:
        is_target = True
    elif which_syl_rpts == 2 and syl2.lower() in target_syllables:
        is_target = True
    
    if transition_probs[idx_1, idx_2] > 0.05 and repeats_list and is_target:
        valid_pairs[pair] = repeats_list

# Step 2: Group pairs by which syllable repeats and apply figure casing
grouped_pairs = defaultdict(list)
for pair, repeats_list in valid_pairs.items():
    if which_syl_rpts == 1:
        name_pair = (pair[0].upper(), pair[1].lower())
        key = pair[0].upper()
    else:
        name_pair = (pair[0].lower(), pair[1].upper())
        key = pair[1].upper()
    grouped_pairs[key].append((name_pair, repeats_list))


############################
# Ask user if they want to save any specific figure
print("\n" + "="*60)
save_fig_choice = input("Would you like to save any specific figure? (yes/no): ").strip().lower()

if save_fig_choice in ['yes', 'y']:
    # Get the target syllable
    target_syl = input("Enter the target syllable (e.g., 'a', 'u', 'g'...): ").strip().lower()
    
    # Check if the syllable exists in our figures
    if target_syl not in figures_dict:
        print(f"[ERROR] No figure found for syllable '{target_syl}'. Available syllables: {list(figures_dict.keys())}")
    else:
        # Get the syllable pairs for this syllable from grouped_by_syllable_end
        syl_upper = target_syl.upper()
        syllable_pairs = grouped_by_syllable_end.get(syl_upper, [])
        
        if not syllable_pairs:
            print(f"[ERROR] No data found for syllable '{target_syl}'")
        else:
            # Calculate max_repeat and collect all repeats
            max_repeat = 0
            all_repeats_combined = []
            for pair, repeats_list in syllable_pairs:
                if repeats_list:
                    max_repeat = max(max_repeat, max(repeats_list))
                    all_repeats_combined.extend(repeats_list)
            
            # Override max_repeat for bird 1, syllable e
            if bird_id == "1" and target_syl.lower() == "e":
                max_repeat = 16

            # Bird 1 syllables e and u: yellow colour, y-axis to 0.8
            if bird_id == "1" and target_syl.lower() == "e" and which_syl_rpts == 2:
                _save_color = "#e69f00"
                _save_y_max = 0.8
            elif bird_id == "1" and target_syl.lower() == "u" and which_syl_rpts == 1:
                _save_color = "#4682b4"
                _save_y_max = 1.0
            else:
                _save_color = "#d3d3d3"
                _save_y_max = 1.0
            
            # Calculate percentages
            percentages = calculate_percentages_for_syllable_group(repeats_by_pair, target_syl, which_syl_rpts)
            
            # Create new figure with "all contexts" + individual contexts
            num_cols = len(syllable_pairs) + 1
            fig_save, axes_save = plt.subplots(1, num_cols, figsize=(num_cols * 6, 5), sharey=True)
            if num_cols == 1:
                axes_save = [axes_save]
            
            # Plot "all contexts" in leftmost panel (no title)
            plot_repeat_distribution_relative(
                all_repeats_combined, 
                (target_syl, target_syl),
                axes_save[0], 
                which_syl_rpts,
                color=_save_color,
                pct_text="",
                max_repeat=max_repeat,
                y_max=_save_y_max,
            )
            axes_save[0].set_title("", fontsize=28)
            
            # Plot individual contexts
            for ax_counter, (pair, repeats_list) in enumerate(syllable_pairs):
                pct_text = ""
                if pair in percentages:
                    pct_text = f" ({percentages[pair]:.1f}%)"
                
                plot_repeat_distribution_relative(
                    repeats_list, 
                    pair, 
                    axes_save[ax_counter + 1],
                    which_syl_rpts,
                    color=_save_color,
                    pct_text=pct_text,
                    max_repeat=max_repeat,
                    y_max=_save_y_max,
                )
            
            plt.tight_layout(pad=2.0, w_pad=3.0)
            plt.subplots_adjust(top=0.92, left=0.18, right=0.98, bottom=0.15)
        
        # Determine the filename based on which syllable repeats
        if which_syl_rpts == 1:
            # Upcoming context: first syllable repeats (e.g., Ag, Ah, Au)
            filename = f"bird_{bird_id}_{target_syl.upper()}_upcoming_contexts.png"
        else:
            # Preceding context: second syllable repeats (e.g., gA, hA, uA)
            filename = f"bird_{bird_id}_{target_syl.upper()}_preceding_contexts.png"
        
        # Define save path in figures/Figure 1
        from pathlib import Path
        save_dir = Path(repo_root) / "figures" / "Figure 1"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        
        # Save the figure
        try:
            fig_save.savefig(save_path, dpi=300, bbox_inches='tight')
            fig_save.savefig(save_path.with_suffix('.svg'), bbox_inches='tight')
            print(f"[INFO] Figure saved successfully at:\n{save_path}")
            plt.close(fig_save)
        except Exception as e:
            print(f"[ERROR] Failed to save figure: {e}")
else:
    print("[INFO] No figure saved.")
