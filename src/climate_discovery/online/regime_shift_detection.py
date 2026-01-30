"""Regime shift detection for climate monitoring.

Detects when ocean regions transition between regimes:
- Historical regime tracking
- Transition alerts
- Spatial extent analysis
- Climate change indicators

Usage:
    from climate_discovery.online import RegimeShiftDetector
    
    detector = RegimeShiftDetector()
    detector.track_historical_regimes(data_2020_2024)
    
    # Monitor new month
    alert = detector.check_for_shifts(data_2025_01)
    if alert:
        print(f"⚠️ {alert.region} shifted Regime {alert.from_id} → {alert.to_id}")
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, NamedTuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RegimeShiftAlert(NamedTuple):
    """Alert for detected regime shift."""
    timestamp: datetime
    region: str
    from_regime: int
    to_regime: int
    n_affected_points: int
    spatial_extent_pct: float
    confidence: float
    description: str


class RegimeShiftDetector:
    """Detect and track ocean regime transitions.
    
    Useful for:
    - Climate change monitoring
    - El Niño/La Niña detection
    - Ecosystem health tracking
    - Early warning systems
    
    Example:
        >>> detector = RegimeShiftDetector(sensitivity=0.15)
        >>> detector.track_historical_regimes(historical_data)
        >>> alerts = detector.check_for_shifts(new_data)
        >>> for alert in alerts:
        ...     print(f"{alert.region}: R{alert.from_regime} → R{alert.to_regime}")
    """
    
    def __init__(
        self,
        sensitivity: float = 0.20,
        min_duration_months: int = 3,
        spatial_threshold: float = 0.10,
    ):
        """Initialize shift detector.
        
        Args:
            sensitivity: Minimum % of region to shift to trigger alert
            min_duration_months: Require shift to persist for N months
            spatial_threshold: Minimum spatial extent to consider significant
        """
        self.sensitivity = sensitivity
        self.min_duration_months = min_duration_months
        self.spatial_threshold = spatial_threshold
        
        # Historical regime assignments
        self.historical_regimes = {}  # {region: {time: regime_labels}}
        self.baseline_regime_fractions = {}  # {region: regime_id -> fraction}
    
    def track_historical_regimes(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        regime_labels: np.ndarray,
        timestamps: np.ndarray,
    ):
        """Build historical regime baseline.
        
        Args:
            lats: Latitudes (N,)
            lons: Longitudes (N,)
            regime_labels: Regime assignments (N,)
            timestamps: Time values (N,)
        """
        logger.info("Building historical regime baseline...")
        
        # Define ocean regions
        regions = self._assign_regions(lats, lons)
        
        # Track regime distribution over time
        unique_times = np.unique(timestamps)
        
        for region_name in np.unique(regions):
            region_mask = regions == region_name
            
            if region_name not in self.historical_regimes:
                self.historical_regimes[region_name] = {}
            
            for t in unique_times:
                time_mask = timestamps == t
                combined_mask = region_mask & time_mask
                
                if np.sum(combined_mask) > 0:
                    self.historical_regimes[region_name][t] = regime_labels[combined_mask]
        
        # Compute baseline regime fractions
        for region_name in self.historical_regimes:
            all_regimes = np.concatenate(list(self.historical_regimes[region_name].values()))
            
            unique_regimes, counts = np.unique(all_regimes, return_counts=True)
            fractions = counts / len(all_regimes)
            
            self.baseline_regime_fractions[region_name] = dict(zip(unique_regimes, fractions))
        
        logger.info(f"✓ Baseline established for {len(self.baseline_regime_fractions)} regions")
    
    def check_for_shifts(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        regime_labels: np.ndarray,
        timestamp: datetime,
    ) -> List[RegimeShiftAlert]:
        """Check for regime shifts in new data.
        
        Args:
            lats: Latitudes
            lons: Longitudes
            regime_labels: Current regime assignments
            timestamp: Current time
            
        Returns:
            List of alerts for detected shifts
        """
        if not self.baseline_regime_fractions:
            logger.warning("No baseline - call track_historical_regimes() first")
            return []
        
        alerts = []
        regions = self._assign_regions(lats, lons)
        
        for region_name in np.unique(regions):
            region_mask = regions == region_name
            region_regimes = regime_labels[region_mask]
            
            if region_name not in self.baseline_regime_fractions:
                continue
            
            # Compute current regime distribution
            unique_regimes, counts = np.unique(region_regimes, return_counts=True)
            current_fractions = dict(zip(unique_regimes, counts / len(region_regimes)))
            
            # Compare to baseline
            baseline = self.baseline_regime_fractions[region_name]
            
            for regime_id, current_frac in current_fractions.items():
                baseline_frac = baseline.get(regime_id, 0.0)
                change = current_frac - baseline_frac
                
                # Significant increase?
                if change > self.sensitivity:
                    # Find what regime it shifted FROM
                    max_decrease = 0
                    from_regime = None
                    
                    for old_id, old_frac in baseline.items():
                        new_frac = current_fractions.get(old_id, 0.0)
                        decrease = old_frac - new_frac
                        
                        if decrease > max_decrease:
                            max_decrease = decrease
                            from_regime = old_id
                    
                    if from_regime is not None and from_regime != regime_id:
                        # Create alert
                        n_affected = int(len(region_regimes) * change)
                        
                        alert = RegimeShiftAlert(
                            timestamp=timestamp,
                            region=region_name,
                            from_regime=from_regime,
                            to_regime=regime_id,
                            n_affected_points=n_affected,
                            spatial_extent_pct=change * 100,
                            confidence=min(change / self.sensitivity, 1.0),
                            description=self._generate_description(
                                region_name, from_regime, regime_id, change
                            ),
                        )
                        
                        alerts.append(alert)
                        logger.warning(f"⚠️ Regime shift detected: {alert.description}")
        
        return alerts
    
    def _assign_regions(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Assign points to ocean regions."""
        regions = np.empty(len(lats), dtype=object)
        
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            # Simplified region assignment
            if lat > 60:
                regions[i] = "Arctic"
            elif lat < -60:
                regions[i] = "Southern Ocean"
            elif -80 <= lon <= 20:
                if lat > 0:
                    regions[i] = "North Atlantic"
                else:
                    regions[i] = "South Atlantic"
            elif 20 < lon <= 150:
                if lat > 0:
                    regions[i] = "North Indian"
                else:
                    regions[i] = "South Indian"
            else:
                if lat > 0:
                    regions[i] = "North Pacific"
                else:
                    regions[i] = "South Pacific"
        
        return regions
    
    def _generate_description(
        self,
        region: str,
        from_regime: int,
        to_regime: int,
        magnitude: float,
    ) -> str:
        """Generate human-readable shift description."""
        # Map regimes to interpretable names (customize based on your model)
        regime_names = {
            0: "Cold Upwelling",
            1: "Warm Oligotrophic",
            2: "Temperate Productive",
            3: "Polar High-CO₂",
            4: "Tropical Low-CO₂",
            5: "Frontal Transition",
        }
        
        from_name = regime_names.get(from_regime, f"Regime {from_regime}")
        to_name = regime_names.get(to_regime, f"Regime {to_regime}")
        
        return (
            f"{region} shifted from {from_name} to {to_name} "
            f"({magnitude*100:.1f}% of region affected)"
        )
    
    def generate_report(
        self,
        alerts: List[RegimeShiftAlert],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate summary report of detected shifts.
        
        Args:
            alerts: List of shift alerts
            save_path: Optional path to save CSV
            
        Returns:
            DataFrame with alert summary
        """
        if not alerts:
            logger.info("No regime shifts detected")
            return pd.DataFrame()
        
        data = []
        for alert in alerts:
            data.append({
                'Timestamp': alert.timestamp,
                'Region': alert.region,
                'From Regime': alert.from_regime,
                'To Regime': alert.to_regime,
                'Points Affected': alert.n_affected_points,
                'Spatial Extent (%)': alert.spatial_extent_pct,
                'Confidence': alert.confidence,
                'Description': alert.description,
            })
        
        df = pd.DataFrame(data)
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"✓ Regime shift report saved: {save_path}")
        
        return df
