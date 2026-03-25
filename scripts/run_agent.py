from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from sdmose.agent import SDMoSEAgent
from reproducibility import (
    enforce_deterministic_runtime,
    log_run_manifest,
    validate_determinism_config,
)


@hydra.main(config_path="../configs", config_name="agent")
def main(cfg):
    cfg = validate_determinism_config(cfg)
    enforce_deterministic_runtime(cfg)
    log_run_manifest(cfg)

    print("Initializing Agent...")
    agent = SDMoSEAgent(cfg)
    agent.run_loop()


if __name__ == "__main__":
    main()
