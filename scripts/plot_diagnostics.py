"""
Agent Diagnostics Plotting

Generates 3 key visualizations:
1. Hypothesis growth trajectory (baseline vs no_memory)
2. Belief entropy comparison across ablations
3. Equation diversity snapshot
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics():
    """Load ablation metrics from JSON."""
    metrics_path = Path("results/ablations/ablation_metrics.json")
    with open(metrics_path) as f:
        return json.load(f)


def plot_hypothesis_growth(metrics, output_dir):
    """Plot 1: Hypothesis count over iterations (baseline vs no_memory)."""
    plt.figure(figsize=(8, 5))

    # Baseline
    baseline = metrics["baseline"]
    plt.plot(
        baseline["iterations"],
        baseline["num_hypotheses"],
        marker="o",
        linewidth=2,
        label="Baseline (with pruning)",
        color="#2E86AB",
    )

    # No memory (no pruning)
    no_mem = metrics["no_memory"]
    plt.plot(
        no_mem["iterations"],
        no_mem["num_hypotheses"],
        marker="s",
        linewidth=2,
        label="No Memory (unbounded)",
        color="#A23B72",
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Hypotheses in Memory", fontsize=12)
    plt.title("Memory Pruning Prevents Hypothesis Explosion", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/hypothesis_growth.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [+] Saved: hypothesis_growth.png")


def plot_entropy_comparison(metrics, output_dir):
    """Plot 2: Final belief entropy across ablations."""
    ablations = ["baseline", "no_verify", "no_memory", "no_belief", "no_reasoning"]
    labels = ["Baseline\n(Full Agent)", "No Verify", "No Memory", "No Belief", "No Reasoning"]

    entropies = [metrics[abl]["belief_entropy"][-1] for abl in ablations]
    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, entropies, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add horizontal line at max entropy
    max_entropy = np.log(3)  # log(num_regimes)
    plt.axhline(
        max_entropy,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Max Entropy (uniform) = {max_entropy:.3f}",
    )

    # Annotate bars
    for bar, ent in zip(bars, entropies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{ent:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylabel("Belief Entropy", fontsize=12)
    plt.title(
        "Belief Concentration Requires Verification & Updates", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.ylim(0, 1.2)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/entropy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [+] Saved: entropy_comparison.png")


def create_equation_table(metrics, output_dir):
    """Plot 3: Equation diversity snapshot (as text table)."""
    ablations = ["baseline", "no_verify", "no_memory", "no_belief", "no_reasoning"]

    _fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    table_data.append(["Ablation", "Hypotheses", "Entropy", "Top Equations"])

    for abl in ablations:
        name = abl.replace("_", " ").title()
        num_hyp = metrics[abl]["num_hypotheses"][-1]
        entropy = f"{metrics[abl]['belief_entropy'][-1]:.3f}"

        # Get top 3 unique equations
        eqs = metrics[abl]["equations"]
        unique_eqs = []
        for eq in eqs:
            if eq not in unique_eqs:
                unique_eqs.append(eq)
            if len(unique_eqs) >= 3:
                break
        eq_str = ", ".join(unique_eqs) if unique_eqs else "—"

        table_data.append([name, str(num_hyp), entropy, eq_str])

    # Create table
    table = ax.table(
        cellText=table_data, cellLoc="left", loc="center", colWidths=[0.20, 0.15, 0.15, 0.50]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor("#2E86AB")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#F0F0F0")

    plt.title("Symbolic Interpretability Across Ablations", fontsize=14, fontweight="bold", pad=20)

    plt.savefig(f"{output_dir}/equation_diversity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [+] Saved: equation_diversity.png")


def main():
    """Generate all diagnostic plots."""
    print("\n" + "=" * 60)
    print("GENERATING AGENT DIAGNOSTICS")
    print("=" * 60 + "\n")

    # Load data
    metrics = load_metrics()

    # Create output directory
    output_dir = Path("results/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Creating plots...")
    plot_hypothesis_growth(metrics, output_dir)
    plot_entropy_comparison(metrics, output_dir)
    create_equation_table(metrics, output_dir)

    print("\n" + "=" * 60)
    print(f"DIAGNOSTICS SAVED TO: {output_dir}/")
    print("=" * 60)
    print("\nPlots generated:")
    print("  1. hypothesis_growth.png")
    print("  2. entropy_comparison.png")
    print("  3. equation_diversity.png")


if __name__ == "__main__":
    main()
