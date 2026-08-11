"""
Simple test script for the equation-discovery agent (no Hydra to avoid Python 3.14 incompatibility).
"""

import numpy as np
from omegaconf import OmegaConf

from physics_discovery.core.agent import DiscoveryAgent
from physics_discovery.core.archive import HypothesisArchive


def main():
    # Simple config without Hydra
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "test_agent"},
            "agent": {
                "type": "grail_v",
                "perception": {"encoder": "identity"},
                "memory": {"capacity": 100},
                "learning": {"max_iterations": 5},
                "num_regimes": 3,
            },
        }
    )

    print("=" * 60)
    print("Discovery Agent Test")
    print("=" * 60)

    # Initialize agent
    print("\n[+] Initializing agent with config...")
    agent = DiscoveryAgent(cfg)

    # Initialize memory
    agent.memory = HypothesisArchive()
    print("[+] Memory initialized")

    # Test agent loop structure
    print("\n[+] Testing agent loop structure...")
    print(f"  - observe() exists: {hasattr(agent, 'observe')}")
    print(f"  - retrieve() exists: {hasattr(agent, 'retrieve')}")
    print(f"  - reason() exists: {hasattr(agent, 'reason')}")
    print(f"  - verify() exists: {hasattr(agent, 'verify')}")
    print(f"  - learn() exists: {hasattr(agent, 'learn')}")
    print(f"  - step() exists: {hasattr(agent, 'step')}")

    # Test hypothesis creation
    print("\n[+] Testing Hypothesis class...")
    from engine.expressions.hypothesis import Hypothesis

    hyp = Hypothesis(equation="y = 2*x + 1", regime_id=0)
    print(f"  - Created hypothesis: {hyp}")

    # ** NEW: End-to-end test with step() **
    print("\n[+] Running full agent loop with step()...")

    # Create sample data
    sample_data = {
        "features": np.random.randn(10, 3),
        "targets": np.random.randn(10),
        "metadata": {"time": "t=0"},
    }

    print(
        f"  - Sample data shape: features={sample_data['features'].shape}, targets={sample_data['targets'].shape}"
    )

    # Run one iteration
    agent.step(sample_data)

    # Check results
    print("\n[+] Step complete! Results:")
    print(f"  - Proposed hypotheses: {len(agent.proposed_hypotheses)}")
    print(f"  - Verified hypotheses: {len(agent.verified_hypotheses)}")
    print(
        f"  - Rejected hypotheses: {len(agent.proposed_hypotheses) - len(agent.verified_hypotheses)}"
    )
    print(f"  - Memory size: {len(agent.memory.hypotheses)}")
    print(f"  - Belief state: {agent.belief.beliefs if agent.belief else 'None'}")

    # Show sample hypothesis
    if agent.verified_hypotheses:
        print("\n  Sample verified hypothesis:")
        print(f"    {agent.verified_hypotheses[0]}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Agent architecture test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
