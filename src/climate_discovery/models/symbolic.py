from typing import List, Dict, Any, Optional
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import yaml

class SymbolicDiscovery:
    """Wrapper around PySR for climate equation discovery."""

    def __init__(self, config: Dict[str, Any], feature_cols: List[str] = None, target_col: str = "fCO2"):
        """
        Args:
            config: Dictionary containing PySR configuration parameters.
            feature_cols: List of feature column names. Defaults to ["SST", "Salinity", "Year", "AbsLat"].
            target_col: Name of the target column. Defaults to "fCO2".
        """
        self.config = config
        self.feature_cols = feature_cols or ["SST", "Salinity", "Year", "AbsLat"]
        self.target_col = target_col
        self.model = PySRRegressor(**self.config)

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> float:
        """
        Fits the symbolic regression model to the data.

        Args:
            df: DataFrame containing features and target.
            test_size: Proportion of dataset to include in the test split.
            random_state: Seed for random number generation.

        Returns:
            The R^2 score on the test set.
        """
        X = df[self.feature_cols]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        
        return self.model.score(X_test, y_test)

    def get_best_equation(self) -> str:
        """Returns the string representation of the best discovered equation."""
        return self.model.sympy()

    @classmethod
    def from_yaml(cls, config_path: str) -> "SymbolicDiscovery":
        """Factory method to create an instance from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
