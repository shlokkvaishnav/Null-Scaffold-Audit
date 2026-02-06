"""
GRAIL-V Ablation Experiments

Runs 5 configurations:
- A0: Baseline (full agent)
- A1: No verification
- A2: No memory pruning
- A3: No belief updates
- A4: No reasoning

Records metrics for reviewer validation.
"""

import sys
import numpy as np
import yaml
import json
from pathlib import Path

# Add sdmose to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdmose.agent.agent import SDMoSEAgent


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_sample_data(n_samples=50, n_features=3):
    """Create synthetic climate data for testing."""
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    targets = features[:, 0] + 0.5 * features[:, 1] + np.random.randn(n_samples) * 0.1
    
    return {
        "features": features,
        "targets": targets,
        "metadata": {}
    }


def run_ablation(config_name, max_iters=10):
    """Run single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running Ablation: {config_name}")
    print(f"{'='*60}")
    
    # Load config
    config_path = Path(f"configs/ablations/{config_name}.yaml")
    config = load_config(config_path)
    
    # Initialize agent
    agent = SDMoSEAgent(config)
    
    # Create data
    data = create_sample_data()
    
    # Track metrics
    metrics = {
        "config": config_name,
        "iterations": [],
        "num_hypotheses": [],
        "num_rejected": [],
        "belief_entropy": [],
        "equations": []
    }
    
    # Run iterations
    for i in range(max_iters):
        agent.step(data)
        
        # Record metrics
        state = agent.introspect()
        metrics["iterations"].append(i+1)
        metrics["num_hypotheses"].append(state["num_hypotheses"])
        metrics["num_rejected"].append(state["num_rejections"])
        
        # Debug: Check proposed vs verified
        if i == 0:
            print(f"  [DEBUG] Proposed: {len(agent.proposed_hypotheses)}, Verified: {len(agent.verified_hypotheses)}")
            if len(agent.proposed_hypotheses) > 0:
                print(f"  [DEBUG] First proposed: {agent.proposed_hypotheses[0]}")
        
        # Belief entropy
        if state["belief"]:
            pi = np.array(state["belief"])
            entropy = -np.sum(pi * np.log(pi + 1e-12))
            metrics["belief_entropy"].append(entropy)
        else:
            metrics["belief_entropy"].append(None)
        
        # Sample equations
        if i == max_iters - 1:
            metrics["equations"] = state["top_equations"]
    
    # Summary
    print(f"\n[RESULTS]")
    print(f"  Final hypotheses in memory: {metrics['num_hypotheses'][-1]}")
    print(f"  Total rejections: {metrics['num_rejected'][-1]}")
    print(f"  Final belief entropy: {metrics['belief_entropy'][-1]:.4f}" if metrics['belief_entropy'][-1] else "N/A")
    print(f"  Top equations: {metrics['equations'][:3]}")
    
    return metrics


def save_results(all_metrics, output_dir="results/ablations"):
    """Save ablation results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(f"{output_dir}/ablation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create summary table
    with open(f"{output_dir}/summary_table.csv", "w") as f:
        f.write("Ablation,Final_Hypotheses,Total_Rejections,Final_Entropy,Example_Eq1\n")
        for name, metrics in all_metrics.items():
            h = metrics["num_hypotheses"][-1]
            r = metrics["num_rejected"][-1]
            e = metrics["belief_entropy"][-1] if metrics["belief_entropy"][-1] else "N/A"
            eq = metrics["equations"][0] if metrics["equations"] else "None"
            f.write(f"{name},{h},{r},{e},{eq}\n")
    
    print(f"\n[+] Results saved to {output_dir}/")


def main():
    """Run all ablation experiments."""
    ablations = [
        "baseline",
        "no_verify",
        "no_memory",
        "no_belief",
        "no_reasoning"
    ]
    
    all_metrics = {}
    
    for ablation in ablations:
        try:
            metrics = run_ablation(ablation, max_iters=10)
            all_metrics[ablation] = metrics
        except Exception as e:
            print(f"[ERROR] Ablation {ablation} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    save_results(all_metrics)
    
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
