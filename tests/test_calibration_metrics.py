import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import unittest
import numpy as np

from sdmose.utils.metrics import calibration_diagnostics
from scripts.reproduce_benchmarks import run


class TestCalibrationMetrics(unittest.TestCase):
    def test_calibration_diagnostics_values(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        )

        metrics = calibration_diagnostics(y_true, y_prob, n_bins=5)

        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("nll", metrics)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertGreaterEqual(metrics["brier"], 0.0)
        self.assertGreaterEqual(metrics["nll"], 0.0)

    def test_benchmark_runner_exports_calibration_outputs(self):
        outputs = run(Path("configs/paper/benchmark_minimal.yaml"))

        per_seed_path = Path(outputs["calibration_per_seed"])
        aggregate_path = Path(outputs["calibration_aggregate"])
        json_path = Path(outputs["output_json"])

        self.assertTrue(per_seed_path.exists())
        self.assertTrue(aggregate_path.exists())
        self.assertTrue(json_path.exists())

        artifact = json.loads(json_path.read_text())
        self.assertIn("calibration_diagnostics", artifact)
        self.assertIn("per_seed", artifact["calibration_diagnostics"])
        self.assertIn("aggregate", artifact["calibration_diagnostics"])


if __name__ == "__main__":
    unittest.main()
