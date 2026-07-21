"""Integration tests for the FastAPI equation-discovery service."""

from __future__ import annotations

import io
import time
import unittest

import numpy as np
from fastapi.testclient import TestClient

from equation_discovery.api.main import app


def _make_synthetic_csv_bytes(seed: int = 0, n_rows: int = 50) -> bytes:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_rows)
    x1 = rng.normal(size=n_rows)
    y = 2.0 * x0 + 3.0 * x1 + 0.01 * rng.normal(size=n_rows)

    lines = ["x0,x1,y"]
    for a, b, c in zip(x0, x1, y):
        lines.append(f"{a},{b},{c}")
    return ("\n".join(lines) + "\n").encode("utf-8")


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("docs", response.json())

    def test_full_discovery_flow(self):
        csv_bytes = _make_synthetic_csv_bytes()
        upload_response = self.client.post(
            "/datasets",
            files={"file": ("synthetic.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"target_column": "y"},
        )
        self.assertEqual(upload_response.status_code, 200, upload_response.text)
        dataset_info = upload_response.json()
        self.assertEqual(dataset_info["n_rows"], 50)
        self.assertEqual(set(dataset_info["feature_names"]), {"x0", "x1"})
        dataset_id = dataset_info["id"]

        # GET dataset info
        get_dataset_response = self.client.get(f"/datasets/{dataset_id}")
        self.assertEqual(get_dataset_response.status_code, 200)

        # Submit a job
        job_response = self.client.post("/jobs", json={"dataset_id": dataset_id})
        self.assertEqual(job_response.status_code, 200, job_response.text)
        job_id = job_response.json()["id"]

        # Poll until done (BackgroundTasks in TestClient run synchronously
        # after the request completes, but we still poll defensively).
        deadline = time.time() + 10
        result = None
        while time.time() < deadline:
            poll_response = self.client.get(f"/jobs/{job_id}")
            self.assertEqual(poll_response.status_code, 200)
            result = poll_response.json()
            if result["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "done", result)
        self.assertIsNotNone(result["equation"])
        self.assertIsNotNone(result["rmse"])
        self.assertTrue(np.isfinite(result["rmse"]))

    def test_get_unknown_job_returns_404(self):
        response = self.client.get("/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_upload_non_csv_returns_4xx(self):
        response = self.client.post(
            "/datasets",
            files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")},
        )
        self.assertGreaterEqual(response.status_code, 400)
        self.assertLess(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
