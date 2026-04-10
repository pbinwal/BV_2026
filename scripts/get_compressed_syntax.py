"""
get_compressed_syntax.py
========================
Compress song syntax by collapsing consecutive repeated syllables.

For example, 'aaabbbbcc' becomes 'a3b4c2'. This is used to represent the
hierarchical structure of birdsong in a more compact form.

Usage:
    # As a module
    from get_compressed_syntax import compress_syllables, load_and_process_data
    df = load_and_process_data("1")

    # As a standalone script
    python get_compressed_syntax.py
"""

import os
import re
import sys
from get_repeat_data import RepeatDataProcessor, syllables_mapping

# Root of the repository (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compress_syllables(song):
    # Compress syllables like aaabbbbcc -> a3b4c2
    compressed = ''.join([f"{match.group(0)[0]}{len(match.group(0))}" for match in re.finditer(r'(.)\1*', song)])
    return compressed


def load_and_process_data(bird_num):
    """
    Load data for the given bird number, compress song syntax and return processed DataFrame.
    """
    yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{bird_num}.yaml")

    processor = RepeatDataProcessor(bird_num, yaml_path)
    df = processor.get_dataframe()

    df_copy = df.copy()
    df_copy['compressed_song'] = df_copy['song'].apply(compress_syllables)

    return df_copy


if __name__ == "__main__":
    available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
    bird_num = input(f"The available birds are {available_birds}\nEnter bird number (1-6): ").strip()

    if bird_num not in syllables_mapping:
        print("Invalid bird number!")
        sys.exit(1)

    # Validate bird syllables
    if bird_num == "1":
        unique_syllables = ['a', 'u', 'g', 'h', 'e', 'b']
    elif bird_num == "2":
        unique_syllables = ['b', 'c', 'e']
    elif bird_num == "3":
        unique_syllables = ['b', 'c', 'd', 'e']
    elif bird_num == "4":
        unique_syllables = ['b', 'e', 'k']
    elif bird_num == "5":
        unique_syllables = ['b', 'e', 'f']
    elif bird_num == "6":
        unique_syllables = ['a', 'b', 'e', 'f', 'h', 'i', 'm']
    else:
        print("Invalid bird number!")
        sys.exit(1)

    df_processed = load_and_process_data(bird_num)
    print(df_processed.head())
