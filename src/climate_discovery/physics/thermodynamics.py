from typing import Union

import numpy as np


# Physics Constraints
def is_physically_valid(prediction_array: np.ndarray) -> bool:
    """
    Checks if the AI's predicted CO2 values are within the realm of reality.

    Args:
        prediction_array: Array of predicted fCO2 values.

    Returns:
        True if valid, False otherwise.
    """
    # Negative CO2 concentrations are not possible
    if np.any(prediction_array < 0):
        return False

    # Upper limit for CO2 concentration in ppm (safe guard)
    if np.any(prediction_array > 5000):
        return False

    return True


def calculate_henrys_law_constant(
    T_kelvin: Union[float, np.ndarray], Salinity: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculates Henry's Law constant for CO2 in seawater.

    Args:
        T_kelvin: Temperature in Kelvin.
        Salinity: Salinity in PSU.

    Returns:
        Henry's law constant (Ko).
    """
    # Constants for Weiss equation
    A1, A2, A3 = -60.2409, 93.4517, 23.3585
    B1, B2, B3 = 0.023517, -0.023656, 0.0047036

    T_100 = T_kelvin / 100.0

    ln_Ko = (
        A1
        + A2 / T_100
        + A3 * np.log(T_100)
        + Salinity * (B1 + B2 * T_100 + B3 * T_100**2)
    )

    return np.exp(ln_Ko)
