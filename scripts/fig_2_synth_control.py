"""fig_2_synth_control.py

For a single bird, computes Spearman correlations between repeat counts of all
syllable pairs as a function of phrase distance (how many phrases apart they
appear in the song), then validates those correlations against a synthetic null
distribution (Figure 1G analysis).

Workflow
--------
1. Observed correlations
   For every ordered syllable pair (syl1, syl2) and each phrase distance n,
   collect the repeat counts of syl1 and syl2 from all songs where the two
   phrases are exactly n phrases apart. Compute the Spearman correlation and
   p-value across songs. Only pairs with >= 100 data points are included.

2. Synthetic null distribution
   Generate 100 synthetic datasets that preserve the phrase order and sequence
   structure of the real data but resample each phrase's repeat count from its
   empirical repeat-number distribution. 

3. Z-test
   For each observed correlation that is nominally significant (p < 0.05),
   compare it against the mean and SD of the 100 synthetic correlations using a
   one-sided z-test. Positive correlations use a right-tailed test; negative
   correlations use a left-tailed test.

Outputs (saved to scripts/output/Correlations by distance z corrected/):
  bird_{n}_dist_by_phrase_vs_corrs_with_syl.csv
      Observed Spearman correlations: Bird_ID, Syllable_Pair, Distance,
      Correlation, p_value, Sample_size
  bird_{n}_dist_by_phrase_vs_corrs_with_syl_ztest.csv
      Z-test results for nominally significant pairs: same columns plus z,
      ztest_p, Sample_size

These CSVs are read by fig_2_g.py to produce Figure 2G after pooling all birds
and applying Benjamini-Hochberg corrections.

Run once per bird. Typically takes several minutes due to the 100 synthetic
iterations.
"""

import os
import re
import sys
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr

from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT, syllables_mapping

# --- User input ---
available = ", ".join(syllables_mapping.keys())
bird_num = input(f"Available birds: {available}\nEnter bird number: ").strip()
if bird_num not in syllables_mapping:
    print(f"Invalid bird number. Available: {available}")
    sys.exit()

# --- Load YAML config ---
yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")
with open(yaml_path, "r") as f:
    bird_config = yaml.safe_load(f)

out_dir = os.path.join(REPO_ROOT, bird_config["correlations_by_distance_dir"] + " z corrected")
os.makedirs(out_dir, exist_ok=True)

# --- Syllable set ---
unique_syllables = syllables_mapping[bird_num]

# --- Load data ---
df = load_and_process_data(bird_num)
df = df.reset_index(drop=True)

# --- Collect all syllables present (for synthetic repeat distributions) ---
syllables_in_data = set()
for song in df["compressed_song"]:
    for syl, _ in re.findall(r"([a-zA-Z])(\d+)", song):
        syllables_in_data.add(syl)

# ── Helper: collect repeat-count pairs by phrase distance ────────────────────

def get_syl_pairs_by_phrase_dist(input_str, first_syll, second_syll):
    """Return {distance: {key: [repeat_counts]}} for one song."""
    tokens = re.findall(r"([a-zA-Z])(\d+)", input_str)
    key_a = f"rpt#{first_syll}"
    key_b = f"rpt#{second_syll}"

    if first_syll == second_syll:
        key_i = f"rpt#{first_syll}_i"
        key_j = f"rpt#{first_syll}_j"
        results = defaultdict(lambda: {key_i: [], key_j: []})
    else:
        results = defaultdict(lambda: {key_a: [], key_b: []})

    selfpair_count = 0
    for i, (syl_a, num_a) in enumerate(tokens):
        if syl_a != first_syll:
            continue
        for j in range(i + 1, len(tokens)):
            syl_b, num_b = tokens[j]
            if first_syll == second_syll:
                if syl_b != first_syll:
                    continue
                intervening = [tokens[m][0] for m in range(i + 1, j)]
                if first_syll in intervening:
                    continue
                count_between = j - i - 1
                key_i = f"rpt#{first_syll}_i"
                key_j = f"rpt#{first_syll}_j"
                if key_i not in results[count_between]:
                    results[count_between][key_i] = []
                    results[count_between][key_j] = []
                results[count_between][key_i].append(int(num_a))
                results[count_between][key_j].append(int(num_b))
                selfpair_count += 1
            else:
                if syl_b == first_syll:
                    break
                if syl_b == second_syll:
                    count_between = j - i - 1
                    results[count_between][key_a].append(int(num_a))
                    results[count_between][key_b].append(int(num_b))
    return dict(results), selfpair_count, False


def combine_results(df, first_syll, second_syll):
    if first_syll == second_syll:
        key_i = f"rpt#{first_syll}_i"
        key_j = f"rpt#{first_syll}_j"
        master = defaultdict(lambda: {key_i: [], key_j: []})
    else:
        key_a = f"rpt#{first_syll}"
        key_b = f"rpt#{second_syll}"
        master = defaultdict(lambda: {key_a: [], key_b: []})

    total_selfpair_count = 0
    for song in df["compressed_song"]:
        result_dict, selfpair_count, _ = get_syl_pairs_by_phrase_dist(song, first_syll, second_syll)
        total_selfpair_count += selfpair_count
        for dist, values in result_dict.items():
            for key in values.keys():
                if key not in master[dist]:
                    master[dist][key] = []
                master[dist][key].extend(values[key])
    return dict(master), total_selfpair_count


def extract_pair_indices(input_str, first_syll, second_syll):
    """Return {distance: [(i, j)]} indices without repeat values (for caching)."""
    tokens = re.findall(r"([a-zA-Z])(\d+)", input_str)
    results = defaultdict(list)
    selfpair_count = 0
    for i, (syl_a, _) in enumerate(tokens):
        if syl_a != first_syll:
            continue
        for j in range(i + 1, len(tokens)):
            syl_b, _ = tokens[j]
            if first_syll == second_syll:
                intervening = [tokens[m][0] for m in range(i + 1, j)]
                if first_syll in intervening:
                    continue
                count_between = j - i - 1
                results[count_between].append((i, j))
                selfpair_count += 1
            else:
                if syl_b == first_syll:
                    break
                if syl_b == second_syll:
                    count_between = j - i - 1
                    results[count_between].append((i, j))
    return dict(results), selfpair_count


# ── Pre-compute pair indices (re-used across synthetic iterations) ────────────

pair_index_cache = {}
for syl1, syl2 in product(unique_syllables, repeat=2):
    cache = []
    for song_idx, song in enumerate(df["compressed_song"]):
        indices, _ = extract_pair_indices(song, syl1, syl2)
        cache.append((song_idx, indices))
    pair_index_cache[(syl1, syl2)] = cache


def extract_repeat_values_from_cache(synth_df, first_syll, second_syll, cache_entry):
    """Use pre-computed indices to extract repeat values from a synthetic dataset."""
    key_a = f"rpt#{first_syll}"
    key_b = f"rpt#{second_syll}"
    if first_syll == second_syll:
        key_i = f"rpt#{first_syll}_i"
        key_j = f"rpt#{first_syll}_j"
        results = defaultdict(lambda: {key_i: [], key_j: []})
    else:
        results = defaultdict(lambda: {key_a: [], key_b: []})

    for song_idx, indices_dict in cache_entry:
        song = synth_df.iloc[song_idx]["compressed_song"]
        tokens = re.findall(r"([a-zA-Z])(\d+)", song)
        for dist, pairs in indices_dict.items():
            for i, j in pairs:
                if first_syll == second_syll:
                    results[dist][key_i].append(int(tokens[i][1]))
                    results[dist][key_j].append(int(tokens[j][1]))
                else:
                    results[dist][key_a].append(int(tokens[i][1]))
                    results[dist][key_b].append(int(tokens[j][1]))
    return dict(results)


# ── Step 1: Observed correlations ────────────────────────────────────────────

DISTANCE_THRESHOLD = 20
SAMPLE_MIN = 100

obs_rows = []
obs_corrs = {}

for syl1, syl2 in product(unique_syllables, repeat=2):
    result, _ = combine_results(df, syl1, syl2)
    if syl1 == syl2:
        key1, key2 = f"rpt#{syl1}_i", f"rpt#{syl1}_j"
    else:
        key1, key2 = f"rpt#{syl1}", f"rpt#{syl2}"
    for dist in sorted(result.keys()):
        x_vals = result[dist].get(key1, [])
        y_vals = result[dist].get(key2, [])
        if len(x_vals) >= SAMPLE_MIN and len(set(x_vals)) > 1 and len(set(y_vals)) > 1:
            corr, pval = spearmanr(x_vals, y_vals)
            obs_corrs[(syl1, syl2, dist)] = (corr, pval, len(x_vals))
            obs_rows.append([f"bird_{bird_num}", f"{syl1}->{syl2}", dist, corr, pval, len(x_vals)])

obs_df = pd.DataFrame(obs_rows, columns=["Bird_ID", "Syllable_Pair", "Distance", "Correlation", "p_value", "Sample_size"])
obs_df = obs_df[obs_df["Distance"] <= DISTANCE_THRESHOLD]
print(f"Observed: {len(obs_df)} correlation rows")

# ── Step 2: Identify nominally significant pairs to z-test ───────────────────

sig_keys = [k for k, v in obs_corrs.items() if v[1] < 0.05 and v[2] >= SAMPLE_MIN]
print(f"Nominally significant pairs to z-test: {len(sig_keys)}")

# ── Step 3: Build repeat-number probability distributions ────────────────────

def get_rpt_num_prob(repeat_num_list):
    total = len(repeat_num_list)
    counts = {}
    for n in repeat_num_list:
        counts[n] = counts.get(n, 0) + 1
    return {n: c / total for n, c in sorted(counts.items())}


def get_syllable_repeats(df, syllable):
    repeats = []
    for song in df["compressed_song"]:
        repeats.extend(int(m) for m in re.findall(rf"{syllable}(\d+)", song))
    return repeats


rpt_prob_dict = {}
for syl in syllables_in_data:
    rpt_prob_dict[syl] = get_rpt_num_prob(get_syllable_repeats(df, syl))

# ── Step 4: Synthetic simulations ────────────────────────────────────────────

N_SYNTH = 100
synth_corrs = {k: [] for k in sig_keys}

for iteration in range(N_SYNTH):
    print(f"Synthetic iteration {iteration + 1}/{N_SYNTH}")
    np.random.seed(42 + iteration)
    synth_df = df.copy()
    synth_songs = []
    for song in synth_df["compressed_song"]:
        new_song = ""
        for syl, _ in re.findall(r"([a-zA-Z])(\d+)", song):
            if syl not in rpt_prob_dict:
                continue
            values = list(rpt_prob_dict[syl].keys())
            probs  = list(rpt_prob_dict[syl].values())
            sample = np.random.choice(values, size=1, p=probs)[0]
            new_song += syl + str(sample)
        synth_songs.append(new_song)
    synth_df["compressed_song"] = synth_songs

    for (syl1, syl2, dist) in sig_keys:
        result = extract_repeat_values_from_cache(synth_df, syl1, syl2, pair_index_cache[(syl1, syl2)])
        key1 = f"rpt#{syl1}_i" if syl1 == syl2 else f"rpt#{syl1}"
        key2 = f"rpt#{syl1}_j" if syl1 == syl2 else f"rpt#{syl2}"
        if dist in result:
            x_vals = result[dist].get(key1, [])
            y_vals = result[dist].get(key2, [])
            if len(x_vals) >= 10 and len(set(x_vals)) > 1 and len(set(y_vals)) > 1:
                corr, _ = spearmanr(x_vals, y_vals)
                synth_corrs[(syl1, syl2, dist)].append(corr)

# ── Step 5: Z-test ────────────────────────────────────────────────────────────

ztest_rows = []
for k in sig_keys:
    obs_corr, obs_p, sample_size = obs_corrs[k]
    synth_dist = synth_corrs[k]
    if len(synth_dist) > 1:
        mu    = np.mean(synth_dist)
        sigma = np.std(synth_dist, ddof=1)
        if sigma > 0:
            z = (obs_corr - mu) / sigma
            p = (1 - norm.cdf(z)) if obs_corr > 0 else norm.cdf(z)
        else:
            z, p = np.nan, np.nan
    else:
        z, p = np.nan, np.nan
    ztest_rows.append([f"bird_{bird_num}", f"{k[0]}->{k[1]}", k[2], obs_corr, obs_p, z, p, sample_size])

ztest_df = pd.DataFrame(ztest_rows, columns=["Bird_ID", "Syllable_Pair", "Distance", "Correlation", "p_value", "z", "ztest_p", "Sample_size"])
ztest_df = ztest_df[ztest_df["Distance"] <= DISTANCE_THRESHOLD]

# ── Save ──────────────────────────────────────────────────────────────────────

obs_out   = os.path.join(out_dir, f"bird_{bird_num}_dist_by_phrase_vs_corrs_with_syl.csv")
ztest_out = os.path.join(out_dir, f"bird_{bird_num}_dist_by_phrase_vs_corrs_with_syl_ztest.csv")
obs_df.to_csv(obs_out,   index=False)
ztest_df.to_csv(ztest_out, index=False)
print(f"\nSaved: {obs_out}")
print(f"Saved: {ztest_out}")
