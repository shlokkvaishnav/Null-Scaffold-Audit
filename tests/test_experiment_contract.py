import unittest

from engine.experiments.contract import (
    BASELINE_EXPERIMENT_CONTRACT,
    ExperimentContractError,
    validate_baseline_contract,
)


class TestExperimentContract(unittest.TestCase):
    def test_valid_contract_passes(self):
        config = {
            **BASELINE_EXPERIMENT_CONTRACT,
            "models": ["pysr_global", "discovery_agent_full"],
            "ablations": ["no_scoring"],
        }
        validate_baseline_contract(config, runner_name="test")

    def test_contract_violation_fails(self):
        config = {
            **BASELINE_EXPERIMENT_CONTRACT,
            "models": ["pysr_global"],
            "metrics": ["rmse", "mae"],
        }

        with self.assertRaises(ExperimentContractError):
            validate_baseline_contract(config, runner_name="test")


if __name__ == "__main__":
    unittest.main()
