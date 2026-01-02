from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm


class RegimeHMM:
    """Wrapper for Hidden Markov Model to detect ocean physics regimes."""

    def __init__(self, n_states: int = 2, random_state: int = 42):
        """
        Args:
            n_states: Number of hidden states (regimes).
            random_state: Seed for reproducibility.
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_state,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the HMM and predicts the hidden state sequence.

        Args:
            X: Input data array (n_samples, n_features).

        Returns:
            Array of predicted hidden states.
        """
        self.model.fit(X)
        return self.model.predict(X)

    @staticmethod
    def plot_regimes(
        df: pd.DataFrame,
        states: np.ndarray,
        x_col: str,
        y_col: str,
        save_path: Optional[str] = None,
    ):
        """
        Plots the time series colored by regime.

        Args:
            df: DataFrame containing the data.
            states: Array of state labels.
            x_col: Name of x-axis column (time).
            y_col: Name of y-axis column (value).
            save_path: If provided, saves the plot to this path.
        """
        plt.figure(figsize=(12, 6))

        # Determine colors based on number of states
        cmap = plt.get_cmap("Set1", np.max(states) + 1)

        plt.scatter(df[x_col], df[y_col], c=states, cmap=cmap, s=15, alpha=0.7)
        plt.plot(df[x_col], df[y_col], c="gray", alpha=0.3, linewidth=1)

        plt.title(f"Dynamic Regime Switching")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.colorbar(ticks=range(np.max(states) + 1), label="Regime")

        if save_path:
            plt.savefig(save_path)
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
