"""Scoring a candidate against data, and against a known answer.

Two questions live here, and they are different. `metrics` asks how well a
candidate predicts held-out targets. `equivalence` asks whether it is the same
expression as a reference one, symbolically or numerically -- which a good
error score does not establish and a poor one does not rule out.

Both are arithmetic over arrays and expression trees, so both are domain
independent. What the reference expression *is* comes from a plugin.
"""

from __future__ import annotations

from engine.evaluation.equivalence import check_equivalence
from engine.evaluation.metrics import calibration_error, compute_fit_metrics

__all__ = ["calibration_error", "check_equivalence", "compute_fit_metrics"]
