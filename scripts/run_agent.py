import hydra

from plugins.physics.scaffold.agent import DiscoveryAgent


@hydra.main(config_path="../configs", config_name="agent")
def main(cfg):
    print("Initializing Agent...")
    agent = DiscoveryAgent(cfg)
    agent.run_loop()


if __name__ == "__main__":
    main()
