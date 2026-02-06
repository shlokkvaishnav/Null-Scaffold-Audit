import hydra
from sdmose.models import baselines

@hydra.main(config_path="../configs", config_name="baseline")
def main(cfg):
    print("Running baselines...")

if __name__ == "__main__":
    main()
