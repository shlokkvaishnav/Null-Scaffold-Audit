import hydra
from omegaconf import DictConfig, OmegaConf

from equation_discovery.experiments.contract import validate_baseline_contract


@hydra.main(config_path="../configs/paper", config_name="benchmark_full", version_base=None)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_baseline_contract(config, runner_name="scripts/run_baselines.py")

    models = config["models"] + config.get("ablations", [])
    print("Running baselines with validated contract:")
    for model in models:
        print(f" - {model}")


if __name__ == "__main__":
    main()
