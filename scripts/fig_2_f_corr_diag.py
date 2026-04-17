"""fig_2_f_corr_diag.py

Run for Bird 2 to generate Figure 2f.

Draws the correlation diagram for a single bird using arrow thickness scaled
by |rho| (absoloute value of correlation coefficient) for significant edges (positive / negative), and dashed uniform-width
arrows for non-significant edges.

Reads pair-detail CSVs produced by plot_fig_2_c.py:
  figures/Figure 2/adjacent_piechart_ztest_pairs.csv
  figures/Figure 2/next_piechart_ztest_pairs.csv

Saves figures/Figure 2/fig_2_f_bird_{n}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG2_DIR = os.path.join(REPO_ROOT, "figures", "Figure 2")

EDGE_COLORS = {
    "positive":        "#e74c3c",
    "negative":        "#2980b9",
    "non-significant": "#bdc3c7",
}

bird_id = input("Enter bird number (e.g. 2): ").strip()
bird_num = int(bird_id)

adj_df  = pd.read_csv(os.path.join(FIG2_DIR, "adjacent_piechart_ztest_pairs.csv"))
next_df = pd.read_csv(os.path.join(FIG2_DIR, "next_piechart_ztest_pairs.csv"))

adj_bird  = adj_df[adj_df["bird_num"]  == bird_num]
next_bird = next_df[next_df["bird_num"] == bird_num]

if adj_bird.empty and next_bird.empty:
    print(f"No data found for bird {bird_num}.")
    exit()


def build_and_draw(ax, sub_df, title):
    G = nx.DiGraph()
    for _, row in sub_df.iterrows():
        rho = row["rho"] if "rho" in row and pd.notna(row["rho"]) else None
        G.add_edge(row["syl1"], row["syl2"], category=row["pie_category"], rho=rho)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Fixed positions for bird 2
    if bird_num == 2:
        FIXED_POS = {
            "b": ( 0.0,  1.0),
            "e": ( 2.0,  1.0),
            "c": (-1.0, -0.5),
        }
        for node in G.nodes():
            key = str(node).lower()
            if key in FIXED_POS:
                pos[node] = FIXED_POS[key]

    for u, v, data in G.edges(data=True):
        cat   = data.get("category", "non-significant")
        color = EDGE_COLORS.get(cat, "#bdc3c7")
        rho   = data.get("rho")
        abs_rho = abs(rho) if rho is not None else 0
        is_sig  = cat in ("positive", "negative")
        # Significant: width scales with |rho|; non-significant: fixed thin dashed
        edge_width = abs_rho * 10 * 1.5 / 2 if is_sig else 1.5
        linestyle  = "solid" if is_sig else "dashed"

        if u == v:
            x, y = pos[u]
            arc_body = FancyArrowPatch(
                posA=(x - 0.12, y + 0.18),
                posB=(x + 0.12, y + 0.18),
                connectionstyle="arc3,rad=-2.8",
                arrowstyle="-",
                color=color,
                linewidth=edge_width,
                linestyle=linestyle,
                mutation_scale=15,
                zorder=1,
            )
            ax.add_patch(arc_body)
            arrowhead = FancyArrowPatch(
                posA=(x + 0.10, y + 0.22),
                posB=(x + 0.12, y + 0.18),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                color=color,
                linewidth=edge_width,
                linestyle="solid",
                mutation_scale=15,
                zorder=2,
            )
            ax.add_patch(arrowhead)
            if rho is not None and is_sig:
                ax.text(x, y + 0.65, f"{rho:.2f}", fontsize=7,
                        ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
        else:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            arrow = FancyArrowPatch(
                posA=(x0, y0),
                posB=(x1, y1),
                connectionstyle="arc3,rad=0.2",
                arrowstyle="-|>",
                color=color,
                linewidth=edge_width,
                linestyle=linestyle,
                mutation_scale=15,
                shrinkA=12, shrinkB=12,
                zorder=1,
            )
            ax.add_patch(arrow)
            if rho is not None and is_sig:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                length = np.hypot(dx, dy) or 1
                ox, oy = dy / length * 0.15, -dx / length * 0.15
                ax.text(mx + ox, my + oy, f"{rho:.2f}", fontsize=7,
                        ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightgrey",
                           edgecolors="#2d3436", linewidths=1.5, ax=ax)
    upper_labels = {node: str(node).upper() for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=upper_labels, font_size=11,
                            font_weight="normal", ax=ax)
    ax.set_title(title, fontsize=9, fontweight="normal")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.axis("off")


fig, axes = plt.subplots(2, 1, figsize=(2.2, 3))
build_and_draw(axes[0], adj_bird,  "Adjacent")
build_and_draw(axes[1], next_bird, "Next")

fig.suptitle(f"Bird {bird_id}", fontsize=10, y=1.01)
plt.tight_layout()

os.makedirs(FIG2_DIR, exist_ok=True)
out_path = os.path.join(FIG2_DIR, f"fig_2_f_bird_{bird_id}.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
fig.savefig(out_path.replace('.png', '.svg'), bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
