"""
Simple test script for SD-MoSE agent (no Hydra to avoid Python 3.14 incompatibility).
"""
from sdmose.agent import SDMoSEAgent, AgentMemory
from omegaconf import OmegaConf

def main():
    # Simple config without Hydra
    cfg = OmegaConf.create({
        "experiment": {"name": "test_agent"},
        "agent": {
            "type": "grail_v",
            "perception": {"encoder": "identity"},
            "memory": {"capacity": 100},
            "learning": {"max_iterations": 5}
        }
    })
    
    print("=" * 60)
    print("SD-MoSE Agent Test (GRAIL-V Architecture)")
    print("=" * 60)
    
    # Initialize agent
    print("\n[+] Initializing agent with config...")
    agent = SDMoSEAgent(cfg)
    
    # Initialize memory
    agent.memory = AgentMemory()
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
    from sdmose.agent import Hypothesis
    hyp = Hypothesis(equation="y = 2*x + 1", regime_id=0)
    print(f"  - Created hypothesis: {hyp}")
    
    # Run agent loop
    print("\n[+] Running agent.run_loop()...")
    agent.run_loop()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Agent architecture test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
