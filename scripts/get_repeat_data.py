"""
get_repeat_data.py
==================
Load preprocessed birdsong data from CSV files and provide it as DataFrames.

This script reads song data (syllable labels, onsets, offsets) that has been
exported as CSV files. Each bird's data is stored in
data/bird_N_df.csv and referenced via the corresponding YAML config file
in yamls/bird_N.yaml.

The CSV files contain the following columns:
    - wav_file: name of the original audio file
    - song: string of syllable labels for that file
    - onsets: space-separated onset times (ms)
    - offsets: space-separated offset times (ms)

Usage:
    # As a module
    from get_repeat_data import RepeatDataProcessor
    processor = RepeatDataProcessor("1", "yamls/bird_1.yaml")
    df = processor.get_dataframe()

    # As a standalone script
    python get_repeat_data.py
"""

import os
import pandas as pd
import yaml
import sys

# Map bird number to unique syllables
# NOTE: For bird 2, syllables 'g' and 'a' were originally segmented as a single
# chunk 'a'. In the CSV, 'a' has been expanded to 'ga' in the song labels, but
# the onsets/offsets still reflect the original single-chunk segmentation
# (i.e., there is one onset/offset pair for the 'ga' chunk, not separate ones).
syllables_mapping = {
    "1": ['a', 'u', 'g', 'h', 'e', 'b'],
    "2": ['b', 'c', 'e'],
    "3": ['b', 'c', 'd', 'e', 'f'],
    "4": ['b', 'e', 'k'],
    "5": ['b', 'e', 'f', 'g', 'h', 'm'],
    "6": ['b', 'e', 'h', 'm']
}

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RepeatDataProcessor:
    """Load preprocessed song data from a CSV file specified in a YAML config."""

    def __init__(self, bird_num, yaml_path):
        self.bird_num = bird_num.strip()
        self.yaml_path = yaml_path
        self.df = None

    def load_data(self):
        """Read the bird's CSV file as specified by the data_csv key in the YAML config."""
        with open(self.yaml_path, "r") as file:
            config = yaml.safe_load(file)

        csv_rel_path = config.get("data_csv")
        if csv_rel_path is None:
            print("No data_csv key found in YAML config!")
            sys.exit(1)

        csv_path = os.path.join(REPO_ROOT, csv_rel_path)
        if not os.path.isfile(csv_path):
            print(f"CSV file not found: {csv_path}")
            sys.exit(1)

        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} rows from {csv_path}")

    def get_dataframe(self):
        if self.df is None:
            self.load_data()
        return self.df


if __name__ == "__main__":
    available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
    bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()

    if bird_num not in syllables_mapping:
        print("Invalid bird number!")
        sys.exit(1)

    yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")

    processor = RepeatDataProcessor(bird_num, yaml_path)
    processor.load_data()
    df = processor.get_dataframe()

    print(df["song"])
