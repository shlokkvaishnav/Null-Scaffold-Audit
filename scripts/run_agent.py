import hydra
from sdmose.agent.agent import Agent

@hydra.main(config_path="../configs", config_name="agent")
def main(cfg):
    print("Initializing GRAIL-V Agent...")
    agent = Agent(cfg)
    agent.run_loop()

if __name__ == "__main__":
    main()
