import numpy as np


class PerceptionModule:
    """
    Grounding: converts raw input data into agent observations.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def encode(self, raw_data):
        """
        Convert raw data into structured observations.

        Args:
            raw_data: dict with 'features', 'targets', 'metadata' keys

        Returns:
            dict: Grounded observation with processed features/targets
        """
        # For now, simple passthrough. A real data pipeline
        # is built in a later phase and can plug in here.

        if isinstance(raw_data, dict):
            return {
                "features": raw_data.get("features", np.array([])),
                "targets": raw_data.get("targets", np.array([])),
                "meta": raw_data.get("metadata", {}),
            }
        else:
            # Handle numpy arrays or other formats
            return {"features": raw_data, "targets": None, "meta": {}}
