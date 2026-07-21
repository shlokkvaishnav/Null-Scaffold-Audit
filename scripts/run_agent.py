import hydra
from physics_discovery.core.agent import DiscoveryAgent

@hydra.main(config_path="../configs", config_name="agent")
def main(cfg):
    print("Initializing Agent...")
    agent = DiscoveryAgent(cfg)
    agent.run_loop()

if __name__ == "__main__":
    main()
