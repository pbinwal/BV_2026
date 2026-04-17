"""fig_3_a_coloured_song.py

Plots colour-coded song sequences for a single bird, where each syllable type
is rendered as a coloured rectangle. Songs are stacked vertically (for example 50
randomly selected songs, sorted longest-first).

Output (saved to figures/Figure 3/):
  bird_{N}_colour_coded_song.png
  bird_{N}_colour_coded_song.svg

Run: rompts for bird number (1-6) and number of songs to plot.
"""

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT, syllables_mapping


# Okabe-Ito colour-blind friendly palette
OKABE_ITO_COLORS = {
    'orange':     '#4400aaff',
    'sky_blue':   '#e69f00ff',
    'green':      '#cc79a7ff',
    'yellow':     '#d55e00ff',
    'blue':       '#009e73ff',
    'vermillion': '#F0E442',
    'purple':     '#56b4e9ff',
    'magenta':    '#FF00FF',
}

OKABE_ITO_LIST = list(OKABE_ITO_COLORS.values())


def assign_colors_to_syllables(unique_syllables):
    color_map = {}
    for idx, syllable in enumerate(unique_syllables):
        color_map[syllable] = OKABE_ITO_LIST[idx % len(OKABE_ITO_LIST)]
    return color_map


def _song_length(compressed):
    """Return total syllable count from a compressed song string (e.g. 'a3b2' → 5)."""
    return sum(int(n) for _, n in re.findall(r'([a-zA-Z])(\d+)', str(compressed)))


def plot_color_coded_songs(df, color_map, n_songs=50, song_column='compressed_song',
                           figsize=(2.8, 2.2), font_size=11, bird_num=None):
    """
    Plot colour-coded songs stacked vertically and save to figures/Figure 3/.

    Parameters
    ----------
    df : pd.DataFrame
    color_map : dict  — syllable → hex colour
    n_songs : int     — number of songs to randomly select (default 100)
    song_column : str — column containing compressed song strings (e.g. 'a3b2c1')
    figsize : tuple
    font_size : int
    bird_num : str    — bird number string used as filename prefix
    """
    if len(df) > n_songs:
        df_sample = df.sample(n=n_songs, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)
        n_songs = len(df_sample)

    # Sort by total syllable count (longest song on top)
    df_sample = df_sample.assign(song_length=df_sample[song_column].apply(_song_length))
    df_sample = df_sample.sort_values('song_length', ascending=False).reset_index(drop=True)
    max_song_length = int(df_sample['song_length'].max())

    plt.rcParams.update({
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "svg.fonttype": "none"
    })

    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sample.iterrows():
        song = row[song_column]
        if pd.isna(song):
            continue
        x_pos = 0
        for syllable, n_str in re.findall(r'([a-zA-Z])(\d+)', str(song)):
            count = int(n_str)
            color = color_map.get(syllable, '#CCCCCC')
            if syllable in ['i', 'j'] or (bird_num == '1' and syllable == 'q'):
                color = 'lightgrey'
            rect = plt.Rectangle((x_pos, idx), count, 1, facecolor=color, edgecolor='none', linewidth=0)
            ax.add_patch(rect)
            x_pos += count

    ax.set_ylim(0, n_songs)
    ax.set_xlim(0, max_song_length)
    ax.set_xlabel('Syllable Index', fontsize=font_size)
    ax.set_ylabel('Song Index', fontsize=font_size)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xticks(list(range(0, max_song_length + 1, 20)))

    mapped_syls = set(syllables_mapping.get(bird_num, []))
    legend_elements = []
    for syllable, orig_color in sorted(color_map.items()):
        label = syllable.upper() if syllable in mapped_syls else syllable
        color = orig_color
        if syllable in ['i', 'j'] or (bird_num == '1' and syllable == 'q'):
            color = 'lightgrey'
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none', label=label))
    leg = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1),
                    title='', frameon=False, fontsize=font_size - 4)
    leg._legend_box.align = "left"

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_dir = os.path.join(REPO_ROOT, "figures", "Figure 3")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"bird_{bird_num}" if bird_num else "bird"
    png_path = os.path.join(save_dir, f"{prefix}_colour_coded_song.png")
    svg_path = os.path.join(save_dir, f"{prefix}_colour_coded_song.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"Saved: {png_path}")

    plt.show()


if __name__ == "__main__":
    available = ", ".join(syllables_mapping.keys())
    bird_num = input(f"Enter bird number ({available}): ").strip()

    if bird_num not in syllables_mapping:
        print(f"Invalid bird number: {bird_num}.")
        sys.exit(1)

    df = load_and_process_data(bird_num)

    all_syllables = sorted(set(''.join(df['compressed_song'].dropna().astype(str))
                               .replace('0123456789', '')))
    # keep only letter characters
    all_syllables = [s for s in all_syllables if s.isalpha()]
    print(f"\nLoaded {len(df)} rows for bird {bird_num}")
    print(f"Syllables present: {all_syllables}")

    color_map = assign_colors_to_syllables(all_syllables)

    n_songs_input = input("\nNumber of songs to plot (default 50): ").strip()
    n_songs = int(n_songs_input) if n_songs_input else 50

    plot_color_coded_songs(df, color_map, n_songs=n_songs,
                           song_column='compressed_song',
                           font_size=11,
                           bird_num=bird_num)
