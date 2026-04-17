"""elasticity.py

Computes repeat-number elasticity for each syllable of a single bird.

Elasticity is defined as the OLS regression slope of repeat_number on song
duration, normalised by the ratio mean_repeat / mean_song_dur (i.e. how much
a syllable's repeat count scales with song length relative to what would be
expected from a fixed proportion).

Exposes two functions for use by fig_3_b_elasticity.py:
  build_repeat_df(df)                    — converts raw song data to per-repeat rows
  calculate_repeat_number_elasticity(repeat_df) — computes elasticity per syllable

When run directly, prompts for a bird number and prints/plots results for that bird.
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from matplotlib.ticker import AutoMinorLocator

from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT, syllables_mapping


def build_repeat_df(df):
    """
    Convert raw song sequences into a dataframe where each row is a repeated syllable.

    Columns produced:
    - 'syl': syllable label
    - 'rpt_num': number of consecutive repeats
    - 'song_len': total syllables in the song
    - 'song_id': wav file
    - 'song_dur': duration of the song in seconds
    """
    rows = []
    for _, row in df.iterrows():
        song = row['song']
        song_len = len(song)
        song_id = row['wav_file']

        syllable_runs = [(match.group(0)[0], len(match.group(0)))
                         for match in re.finditer(r'(.)\1*', song)]

        for syl, rpt_num in syllable_runs:
            rows.append({
                'syl': syl,
                'rpt_num': rpt_num,
                'song_len': song_len,
                'song_id': song_id,
            })

    return pd.DataFrame(rows)


def calculate_repeat_number_elasticity(repeat_df):
    """
    Calculate repeat number elasticity for each syllable using OLS regression.
    Returns a DataFrame with columns: syl, b_i, expected_bi, elasticity, n.
    """
    elasticity_results = []
    for syl, group in repeat_df.groupby('syl'):
        if len(group) < 2:
            continue

        X = sm.add_constant(group['song_len'])
        y = group['rpt_num']
        model = sm.OLS(y, X).fit()

        b_i = model.params['song_len']
        mean_rpt = group['rpt_num'].mean()
        mean_song = group['song_len'].mean()
        expected_bi = mean_rpt / mean_song
        elasticity = b_i / expected_bi

        elasticity_results.append({
            'syl': syl,
            'b_i': b_i,
            'expected_bi': expected_bi,
            'elasticity': elasticity,
            'n': len(group)
        })

    return pd.DataFrame(elasticity_results)


def plot_repeat_elasticity(elasticity_df, bird_num):
    """
    Plot stacked histogram of repeat number elasticity values for a single bird.
    """
    syllables = elasticity_df['syl'].tolist()
    hist_data = [elasticity_df[elasticity_df['syl'] == syl]['elasticity'] for syl in syllables]
    colors = plt.cm.tab10.colors[:len(syllables)]

    plt.figure(figsize=(8, 6))
    plt.hist(hist_data, bins=10, stacked=True, edgecolor='black', color=colors, label=syllables)
    plt.xlabel('Elasticity')
    plt.ylabel('Frequency')
    plt.title(f'Repeat Number Elasticity Histogram for Bird {bird_num}')
    plt.legend(title='Syllable')
    plt.show()


if __name__ == "__main__":
    available_birds = ", ".join(syllables_mapping.keys())
    bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()

    if bird_num not in syllables_mapping:
        print("Invalid bird number!")
        sys.exit(1)

    unique_syllables = syllables_mapping[bird_num]

    df = load_and_process_data(bird_num)

    repeat_df = build_repeat_df(df)
    print("Repeat dataframe preview:\n", repeat_df.head())

    repeat_df = repeat_df[repeat_df['syl'].isin(unique_syllables)]
    print("Filtered repeat dataframe:\n", repeat_df.head())

    elasticity_df = calculate_repeat_number_elasticity(repeat_df)
    print("Elasticity results:\n", elasticity_df)

    plot_repeat_elasticity(elasticity_df, bird_num)
