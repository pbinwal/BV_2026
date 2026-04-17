"""plot_trans_diag.py
 Figure 2 E: Run for bird 2
 
Computes and plots the syllable transition matrix and diagram for a single bird.

Notes:
- Bird 2 only (Fig 2E): the 'ga' bigram chunk is merged into a single internal token 'z'
  before computing the transition matrix, so it appears as one node labelled 'ga' in the
  figure. Lone 'g' nodes are removed after the 5% threshold filter.
  Custom node order for the circular diagram: ['z', 'e', 'b', 'c', 'd', 'f']
  (i.e. displayed as: ga, e, b, c, d, f).
- All other birds use alphabetical node ordering.
"""
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

import sys
import os

# Add parent directory (src) to sys.path so 'rptanalysis' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from get_repeat_data import RepeatDataProcessor, syllables_mapping


available_birds = ", ".join([f"bird {num}" for num in syllables_mapping.keys()])
bird_id = input(f"The available bird ids are {available_birds}\nEnter bird ID (1-6): ").strip()

if bird_id not in syllables_mapping:
    print("Invalid bird number !!!")
    sys.exit()

unique_syllables = syllables_mapping.get(bird_id)
if unique_syllables is None:
    print("INVALID Bird Id!!!")
    sys.exit()

# Construct the YAML file path dynamically using the bird_id
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yaml_path = os.path.join(repo_root, "yamls", f"bird_{bird_id}.yaml")

#  Load the YAML (only if present)

cfg = {}                          # empty default
if os.path.isfile(yaml_path):     # <- check first
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

out_dir = cfg.get("transition_dir", "saved_mats")   # default fallback
os.makedirs(out_dir, exist_ok=True)

# Initialize and load data
processor = RepeatDataProcessor(bird_id, yaml_path)
df = processor.get_dataframe()

print(df.head())

# Now the existing code can work with 'df' 
df_copy = df.copy()


#the following code gets the transition matrix with repeats collapsed and then creates a transition diagram


# Custom node order for circular diagram (bird 2 / Fig 2E)
# 'z' represents the merged 'ga' chunk
CUSTOM_SYLLABLE_ORDER = {
    "2": ['z', 'e', 'b', 'c', 'd', 'f'],
}


def calculate_transition_matrix(sequences, custom_order=None):
    """
    Calculates the transition matrix from a list of sequences.

    Parameters:
    - sequences: A pandas Series of sequences (strings).
    - custom_order: Optional list specifying the desired syllable order.

    Returns:
    - transition_matrix: The normalized transition matrix (2D NumPy array).
    - unique_syllables: List of unique syllables in the used order.
    """
    # Collect all unique syllables in order of first appearance
    seen = set()
    all_syllables = []
    for seq in sequences:
        for ch in seq:
            if ch not in seen:
                seen.add(ch)
                all_syllables.append(ch)

    if custom_order:
        # Place custom-order syllables first (matching lower/upper variants),
        # then append any remaining syllables in appearance order.
        placed = set()
        unique_syllables = []
        for s in custom_order:
            for candidate in all_syllables:
                if candidate.lower() == s.lower() and candidate not in placed:
                    unique_syllables.append(candidate)
                    placed.add(candidate)
        for s in all_syllables:
            if s not in placed:
                unique_syllables.append(s)
                placed.add(s)
    else:
        unique_syllables = sorted(set("".join(sequences)))

    syllable_index = {syllable: i for i, syllable in enumerate(unique_syllables)}

    # Initialize transition count matrix
    transition_matrix = np.zeros((len(unique_syllables), len(unique_syllables)))

    for seq in sequences:
        for i in range(len(seq) - 1):
            current_syllable = seq[i]
            next_syllable = seq[i + 1]
            if current_syllable in syllable_index and next_syllable in syllable_index:
                transition_matrix[syllable_index[current_syllable], syllable_index[next_syllable]] += 1

    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums

    return transition_matrix, unique_syllables


# Function to collapse repeated syllables globally and replace them with a capital letter
def collapse_repeats(sequence, global_repeats):
    collapsed_sequence = []
    i = 0

    while i < len(sequence):
        char = sequence[i]
        repeat_count = 1

        # Count consecutive repetitions of the current character
        while i + 1 < len(sequence) and sequence[i + 1] == char:
            repeat_count += 1
            i += 1

        # Add the character to the global repeats if it's repeated
        if repeat_count > 1:
            global_repeats.add(char)

        collapsed_sequence.append(char)
        i += 1

    # Replace all occurrences of global repeating syllables with their uppercase version
    return ''.join([char.upper() if char in global_repeats else char for char in collapsed_sequence])

# Identify global repeating syllables
global_repeats = set()

# First pass to collect globally repeated syllables
df_copy['song'] = df_copy['song'].apply(lambda seq: collapse_repeats(seq, global_repeats))

#  Apply collapsing with the identified repeating syllables
df_copy['song'] = df_copy['song'].apply(lambda seq: collapse_repeats(seq, global_repeats))

# For bird 2: merge 'ga' bigram into single token 'z' before transition matrix
if bird_id == "2":
    df_copy['song'] = df_copy['song'].str.replace('ga', 'z')
    df_copy['song'] = df_copy['song'].str.replace('GA', 'Z')

# Calculate the transition matrix (assuming 'calculate_transition_matrix' is defined elsewhere)
sequences = df_copy['song']

# Calculate the transition matrix
custom_order = CUSTOM_SYLLABLE_ORDER.get(bird_id, None)
transition_probs, unique_syllables_from_data = calculate_transition_matrix(sequences, custom_order=custom_order)

if bird_id == "1":  # br43br75
    # Replace 'Q' with 'I'
    df_copy['song'] = df_copy['song'].str.replace('Q', 'I')
    # Replace 'b' with 'B'
    df_copy['song'] = df_copy['song'].str.replace('b', 'B')

# Create plot labels
if bird_id == "2":
    # Show 'z'/'Z' as 'ga', remove lone 'g' below after 5% filter
    plot_labels = [
        "ga" if s.lower() == "z" else
        s.upper() if s.lower() in ["b", "c", "e"] else
        s.lower()
        for s in unique_syllables_from_data
    ]
else:
    plot_labels = [s.upper() for s in unique_syllables_from_data]

trans_out_dir = cfg.get("transition_dir")  # Read from YAML 
if not trans_out_dir:
    print("WARNING: 'transition_dir' not found in YAML file. Please add it to the configuration :)")
    sys.exit(1)

os.makedirs(trans_out_dir, exist_ok=True)

np.save(os.path.join(trans_out_dir, f"bird_{bird_id}_transition_probs.npy"), transition_probs)
np.save(os.path.join(trans_out_dir, f"bird_{bird_id}_unique_syllables.npy"), np.array(unique_syllables_from_data))



# Filter transitions below 5%
transition_probs[transition_probs < 0.05] = 0

# For bird 2: remove lone 'g' node from matrix and labels as they are very, very rare mislabels (ga is a chunk where g goes only to a)
if bird_id == "2":
    for lone in ["g", "G"]:
        if lone in unique_syllables_from_data:
            idx = unique_syllables_from_data.index(lone)
            transition_probs = np.delete(transition_probs, idx, axis=0)
            transition_probs = np.delete(transition_probs, idx, axis=1)
            unique_syllables_from_data.pop(idx)
            plot_labels.pop(idx)

def plot_combined(matrix, labels, node_size=1500, edge_width=10, min_prob=0.05):
    """
    Side-by-side figure: heatmap (with numbers) on left, transition diagram (no numbers) on right.
    Arrow style: separate unidirectional arrows only.
    """
    # Create a directed graph
    Graph = nx.DiGraph()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= min_prob:
                Graph.add_edge(i, j, weight=matrix[i, j])

    if bird_id == "1":
        node_labels = {i: ("Start" if label == "Q" else label) for i, label in enumerate(labels)}
    else:
        node_labels = {i: label for i, label in enumerate(labels)}

    edge_weights = nx.get_edge_attributes(Graph, "weight")

    node_order = list(range(len(labels)))
    angles = np.linspace(0, 2 * np.pi, len(node_order), endpoint=False)
    positions = {
        node: np.array([np.cos(angle), np.sin(angle)])
        for node, angle in zip(node_order, angles)
    }

    print("\nNode positions (index: label: [x, y]):")
    for node in node_order:
        pos = positions[node]
        print(f"{node}: {labels[node]}: [{pos[0]:.6f}, {pos[1]:.6f}]")

    pos_array = np.array(list(positions.values()))
    center = pos_array.mean(axis=0)

    fig, (ax_heat, ax_diag) = plt.subplots(1, 2, figsize=(9, 4))

    # --- Left: heatmap with numbers ---
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Greys",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Transition Probability'},
        ax=ax_heat,
        linewidths=0.5,
        linecolor='white',
    )
    ax_heat.set_title("Transition Matrix", fontsize=11)
    ax_heat.set_xlabel("Next", fontsize=10)
    ax_heat.set_ylabel("Current", fontsize=10)
    ax_heat.tick_params(labelsize=10)

    # --- Right: transition diagram, no numbers ---
    nx.draw_networkx_nodes(
        Graph, pos=positions, node_size=400,
        node_color="lightgrey", edgecolors="black", linewidths=2, ax=ax_diag
    )
    nx.draw_networkx_labels(Graph, pos=positions, labels=node_labels, font_size=11, font_weight='normal', ax=ax_diag)

    for edge in Graph.edges():
        u, v = edge
        width = edge_weights[edge] * edge_width * 1.5 / 2
        has_reverse = Graph.has_edge(v, u)
        pos_u = np.array(positions[u])
        pos_v = np.array(positions[v])
        cross = np.cross(pos_u - center, pos_v - center)
        if has_reverse:
            rad = (0.3 if cross > 0 else -0.3) if u < v else (-0.3 if cross > 0 else 0.3)
        else:
            rad = 0.15 if cross > 0 else -0.15
        nx.draw_networkx_edges(
            Graph, pos=positions, edgelist=[edge],
            width=width, edge_color="black", style="solid", ax=ax_diag,
            arrows=True, arrowsize=35, arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=2, min_target_margin=2
        )

    ax_diag.axis("off")

    fig.suptitle(f"Bird {bird_id}", fontsize=12, y=1.01)
    plt.tight_layout()

    fig_out_dir = os.path.join(repo_root, "figures", "Figure 2")
    os.makedirs(fig_out_dir, exist_ok=True)
    out_base = os.path.join(fig_out_dir, f"fig_2_e_bird_{bird_id}")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out_base + ".svg", bbox_inches='tight')
    print(f"\nSaved:\n  {out_base}.png")
    plt.show()


plot_combined(
    matrix=transition_probs,
    labels=plot_labels,
    node_size=2000,
    edge_width=7,
    min_prob=0.05,
)