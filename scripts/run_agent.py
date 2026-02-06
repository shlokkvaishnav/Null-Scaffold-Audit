import hydra
from sdmose.agent import SDMoSEAgent

@hydra.main(config_path="../configs", config_name="agent")
def main(cfg):
    print("Initializing GRAIL-V Agent...")
    agent = SDMoSEAgent(cfg)
    agent.run_loop()

if __name__ == "__main__":
    main()
