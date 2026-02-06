import hydra

@hydra.main(config_path="../configs/ablations", config_name="no_soft")
def main(cfg):
    print("Running ablation study...")

if __name__ == "__main__":
    main()
