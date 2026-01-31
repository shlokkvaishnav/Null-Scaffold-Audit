"""Git-tracked equation versioning system.

Automatically saves discovered equations with:
- Git commit hash
- Timestamp
- Full configuration
- Performance metrics
- Reproducibility info

Usage:
    from climate_discovery.utils.equation_version import EquationVersionManager
    
    manager = EquationVersionManager()
    manager.save_equations(
        equations=equations_dict,
        config=config,
        metrics={"test_r2": 0.45, "val_mse": 123.4},
        version="1.0.0",
    )
"""

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class EquationVersionManager:
    """Manage versioned equation storage with Git tracking.
    
    Args:
        equations_dir: Directory for equation files (default: equations/)
        auto_commit: Whether to auto-commit new equation files
        
    Example:
        >>> manager = EquationVersionManager()
        >>> manager.save_equations(
        ...     equations={0: "fCO2 = 349 + 2*SST"},
        ...     config=config,
        ...     metrics={"r2": 0.45},
        ...     version="1.0.0"
        ... )
        >>> # Saved to: equations/sd-mose_v1.0.0_20260130_a3f4b2c.txt
    """
    
    def __init__(
        self,
        equations_dir: str = "equations",
        auto_commit: bool = False,
    ):
        self.equations_dir = Path(equations_dir)
        self.equations_dir.mkdir(exist_ok=True)
        self.auto_commit = auto_commit
    
    def get_git_info(self) -> Dict[str, str]:
        """Get current Git repository info."""
        try:
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            # Get commit hash (short)
            commit_short = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            # Get branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            # Check if dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            is_dirty = len(status) > 0
            
            return {
                "commit": commit,
                "commit_short": commit_short,
                "branch": branch,
                "is_dirty": is_dirty,
                "dirty_indicator": "*" if is_dirty else "",
            }
        except Exception:
            return {
                "commit": "unknown",
                "commit_short": "unknown",
                "branch": "unknown",
                "is_dirty": True,
                "dirty_indicator": "",
            }
    
    def serialize_config(self, config: Any) -> Dict:
        """Convert config to JSON-serializable dict."""
        if is_dataclass(config):
            config_dict = asdict(config)
        elif hasattr(config, "__dict__"):
            config_dict = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith("_")
            }
        else:
            config_dict = dict(config)
        
        # Handle non-serializable types
        cleaned = {}
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                cleaned[key] = value
            elif isinstance(value, (list, tuple)):
                cleaned[key] = list(value)
            elif isinstance(value, dict):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def save_equations(
        self,
        equations: Dict[int, str],
        config: Any,
        metrics: Optional[Dict[str, float]] = None,
        version: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """Save equations with full versioning metadata.
        
        Args:
            equations: Dict of {regime_id: equation_string}
            config: Model configuration
            metrics: Performance metrics (R², MSE, etc.)
            version: Semantic version (e.g., "1.0.0")
            notes: Optional notes about this version
            
        Returns:
            Path to saved equation file
        """
        # Get Git info
        git_info = self.get_git_info()
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create filename
        if version:
            version_str = f"v{version}"
        else:
            version_str = "dev"
        
        filename = (
            f"sd-mose_{version_str}_{timestamp}_"
            f"{git_info['commit_short']}{git_info['dirty_indicator']}.txt"
        )
        filepath = self.equations_dir / filename
        
        # Create content
        content = []
        
        # Header
        content.append("=" * 80)
        content.append("SD-MoSE DISCOVERED EQUATIONS")
        content.append("=" * 80)
        content.append("")
        
        # Version info
        content.append("[VERSION INFO]")
        content.append(f"Version: {version or 'dev'}")
        content.append(f"Date: {date_readable}")
        content.append(f"Git Commit: {git_info['commit']}")
        content.append(f"Git Branch: {git_info['branch']}")
        if git_info['is_dirty']:
            content.append("⚠️  WARNING: Working directory has uncommitted changes!")
        content.append("")
        
        # Metrics
        if metrics:
            content.append("[PERFORMANCE METRICS]")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    content.append(f"{key}: {value:.6f}")
                else:
                    content.append(f"{key}: {value}")
            content.append("")
        
        # Configuration
        content.append("[CONFIGURATION]")
        config_dict = self.serialize_config(config)
        for key, value in sorted(config_dict.items()):
            content.append(f"{key}: {value}")
        content.append("")
        
        # Notes
        if notes:
            content.append("[NOTES]")
            content.append(notes)
            content.append("")
        
        # Equations
        content.append("[DISCOVERED EQUATIONS]")
        content.append("")
        for regime_id, equation in sorted(equations.items()):
            content.append(f"Regime {regime_id}:")
            content.append(f"  {equation}")
            content.append("")
        
        # Footer
        content.append("=" * 80)
        content.append("REPRODUCIBILITY INFO")
        content.append("=" * 80)
        content.append("")
        content.append("To reproduce these results:")
        content.append(f"1. Checkout commit: git checkout {git_info['commit']}")
        content.append("2. Install dependencies: pip install -r requirements.txt")
        content.append("3. Run training with the configuration above")
        content.append("")
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"✓ Equations saved: {filepath}")
        
        # Auto-commit if enabled
        if self.auto_commit and not git_info['is_dirty']:
            self._git_commit_equations(filepath, version)
        
        # Also save JSON version
        self._save_json_version(filepath, equations, config_dict, metrics, git_info)
        
        return filepath
    
    def _save_json_version(
        self,
        txt_path: Path,
        equations: Dict,
        config: Dict,
        metrics: Optional[Dict],
        git_info: Dict,
    ):
        """Save JSON version for programmatic access."""
        json_path = txt_path.with_suffix(".json")
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "git_is_dirty": git_info["is_dirty"],
            "config": config,
            "metrics": metrics or {},
            "equations": equations,
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def _git_commit_equations(self, filepath: Path, version: Optional[str]):
        """Auto-commit equation file to Git."""
        try:
            # Add file
            subprocess.run(
                ["git", "add", str(filepath)],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            
            # Commit
            msg = f"Save discovered equations v{version or 'dev'}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            
            print(f"✓ Auto-committed to Git: {filepath.name}")
        except Exception as e:
            print(f"⚠️  Failed to auto-commit: {e}")
    
    def list_versions(self) -> list[Dict]:
        """List all saved equation versions."""
        versions = []
        
        for json_file in sorted(self.equations_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                versions.append({
                    "file": json_file.stem,
                    "timestamp": data.get("timestamp"),
                    "commit": data.get("git_commit", "")[:7],
                    "metrics": data.get("metrics", {}),
                })
            except Exception:
                continue
        
        return versions
    
    def load_equations(self, version: str) -> Dict:
        """Load equations from a specific version.
        
        Args:
            version: Version string or filename pattern
            
        Returns:
            Dictionary with config, metrics, equations
        """
        # Find matching file
        pattern = f"*{version}*.json"
        matches = list(self.equations_dir.glob(pattern))
        
        if not matches:
            raise FileNotFoundError(f"No equation file matching: {version}")
        
        if len(matches) > 1:
            print(f"⚠️  Multiple matches, using: {matches[0]}")
        
        with open(matches[0]) as f:
            return json.load(f)
    
    def compare_versions(self, version1: str, version2: str):
        """Compare two equation versions side-by-side."""
        data1 = self.load_equations(version1)
        data2 = self.load_equations(version2)
        
        print("=" * 80)
        print("EQUATION VERSION COMPARISON")
        print("=" * 80)
        
        print(f"\nVersion 1: {version1}")
        print(f"Metrics: {data1.get('metrics', {})}")
        print(f"\nVersion 2: {version2}")
        print(f"Metrics: {data2.get('metrics', {})}")
        
        print("\n" + "-" * 80)
        print("EQUATION DIFFERENCES")
        print("-" * 80)
        
        eqs1 = data1.get("equations", {})
        eqs2 = data2.get("equations", {})
        
        all_regimes = set(eqs1.keys()) | set(eqs2.keys())
        
        for regime in sorted(all_regimes, key=lambda x: int(x)):
            eq1 = eqs1.get(regime, "N/A")
            eq2 = eqs2.get(regime, "N/A")
            
            if eq1 != eq2:
                print(f"\nRegime {regime}:")
                print(f"  v1: {eq1}")
                print(f"  v2: {eq2}")


# Backward compatibility alias
EquationVersionControl = EquationVersionManager