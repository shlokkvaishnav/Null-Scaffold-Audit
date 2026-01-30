"""Experiment tracking integration for SD-MoSE.

Supports both Weights & Biases (wandb) and MLflow for:
- Hyperparameter tracking
- Metric logging (loss, R², regime statistics)
- Equation versioning
- Model checkpointing
- Reproducibility

Usage:
    # Initialize tracker
    tracker = ExperimentTracker(backend="wandb", project="sd-mose")
    
    # Log config
    tracker.log_config(config)
    
    # Log metrics during training
    tracker.log_metrics({"train_loss": 0.5, "val_r2": 0.44}, step=10)
    
    # Log equations
    tracker.log_equations(equations_dict, iteration=5)
    
    # Finish
    tracker.finish()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified interface for experiment tracking (wandb or mlflow).
    
    Args:
        backend: "wandb", "mlflow", or "both"
        project: Project name
        name: Run name (optional, auto-generated if None)
        config: Configuration dict to log
        tags: List of tags for organization
        notes: Run description
        dir: Directory for logs/artifacts
        
    Example:
        >>> tracker = ExperimentTracker("wandb", project="sd-mose", name="hierarchical-v1")
        >>> tracker.log_config(config)
        >>> tracker.log_metrics({"loss": 0.5}, step=10)
        >>> tracker.log_equations({"regime_0": "fCO2 = 349 + 2*SST"}, iteration=1)
        >>> tracker.finish()
    """
    
    def __init__(
        self,
        backend: str = "wandb",
        project: str = "sd-mose",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
        offline: bool = False,
    ):
        self.backend = backend.lower()
        self.project = project
        self.name = name
        self.offline = offline
        
        # Initialize backends
        self.wandb_run = None
        self.mlflow_run = None
        
        if self.backend in ["wandb", "both"]:
            self._init_wandb(config, tags, notes, dir)
        
        if self.backend in ["mlflow", "both"]:
            self._init_mlflow(config, tags, notes)
    
    def _init_wandb(
        self,
        config: Optional[Dict],
        tags: Optional[List[str]],
        notes: Optional[str],
        dir: Optional[str],
    ):
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.project,
                name=self.name,
                config=config,
                tags=tags,
                notes=notes,
                dir=dir,
                mode="offline" if self.offline else "online",
                reinit=True,
            )
            logger.info(f"✓ Weights & Biases initialized: {self.wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install: pip install wandb")
            self.backend = self.backend.replace("wandb", "").replace("both", "mlflow")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
    
    def _init_mlflow(
        self,
        config: Optional[Dict],
        tags: Optional[List[str]],
        notes: Optional[str],
    ):
        """Initialize MLflow."""
        try:
            import mlflow
            
            # Set experiment
            mlflow.set_experiment(self.project)
            
            # Start run
            self.mlflow_run = mlflow.start_run(run_name=self.name)
            
            # Log config
            if config:
                for key, value in config.items():
                    try:
                        mlflow.log_param(key, value)
                    except Exception:
                        pass  # Skip non-serializable
            
            # Log tags
            if tags:
                for tag in tags:
                    mlflow.set_tag("tag", tag)
            
            if notes:
                mlflow.set_tag("notes", notes)
            
            logger.info(f"✓ MLflow initialized: run_id={self.mlflow_run.info.run_id}")
        except ImportError:
            logger.warning("mlflow not installed. Install: pip install mlflow")
            self.backend = self.backend.replace("mlflow", "").replace("both", "wandb")
        except Exception as e:
            logger.error(f"Failed to initialize mlflow: {e}")
    
    def log_config(self, config: Union[Dict, Any]):
        """Log configuration/hyperparameters.
        
        Args:
            config: Config dict or dataclass instance
        """
        # Convert dataclass to dict if needed
        if hasattr(config, "__dict__"):
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        else:
            config_dict = config
        
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                wandb.config.update(config_dict, allow_val_change=True)
            except Exception as e:
                logger.warning(f"wandb config update failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                for key, value in config_dict.items():
                    try:
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(key, value)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"mlflow param logging failed: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics.
        
        Args:
            metrics: Dict of {metric_name: value}
            step: Step/iteration number
            commit: Whether to commit immediately (wandb)
        """
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                logger.warning(f"wandb metric logging failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                for key, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning(f"mlflow metric logging failed: {e}")
    
    def log_equations(
        self,
        equations: Dict[int, str],
        iteration: Optional[int] = None,
        prefix: str = "equation",
    ):
        """Log discovered equations.
        
        Args:
            equations: Dict of {regime_id: equation_string}
            iteration: SD-MoSE iteration number
            prefix: Prefix for equation names
        """
        # Create equations text
        equations_text = "\n".join([
            f"Regime {k}: {eq}" for k, eq in equations.items()
        ])
        
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                
                # Log as text
                wandb.log({
                    f"{prefix}_text": wandb.Html(f"<pre>{equations_text}</pre>"),
                }, step=iteration)
                
                # Also log each equation separately
                for k, eq in equations.items():
                    wandb.log({f"{prefix}_{k}": eq}, step=iteration)
                
            except Exception as e:
                logger.warning(f"wandb equation logging failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                
                # Log as artifact
                equations_path = Path(f"equations_iter_{iteration}.txt")
                with open(equations_path, "w") as f:
                    f.write(equations_text)
                mlflow.log_artifact(str(equations_path))
                equations_path.unlink()  # Clean up
                
            except Exception as e:
                logger.warning(f"mlflow equation logging failed: {e}")
    
    def log_artifact(
        self,
        path: Union[str, Path],
        artifact_type: Optional[str] = None,
    ):
        """Log file artifact (checkpoint, figure, etc).
        
        Args:
            path: Path to file
            artifact_type: Type of artifact (for wandb)
        """
        path = Path(path)
        
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                wandb.save(str(path), base_path=path.parent)
            except Exception as e:
                logger.warning(f"wandb artifact logging failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(str(path))
            except Exception as e:
                logger.warning(f"mlflow artifact logging failed: {e}")
    
    def log_model(
        self,
        model_path: Union[str, Path],
        model_name: str = "sd-mose",
    ):
        """Log PyTorch model checkpoint.
        
        Args:
            model_path: Path to .pth file
            model_name: Name for model artifact
        """
        self.log_artifact(model_path, artifact_type="model")
        
        # MLflow: Also register in model registry
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(str(model_path), artifact_path="models")
            except Exception as e:
                logger.warning(f"mlflow model logging failed: {e}")
    
    def log_figure(
        self,
        figure,
        name: str,
        step: Optional[int] = None,
    ):
        """Log matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            name: Figure name
            step: Step number
        """
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                wandb.log({name: wandb.Image(figure)}, step=step)
            except Exception as e:
                logger.warning(f"wandb figure logging failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                from io import BytesIO
                
                buf = BytesIO()
                figure.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                
                # Save temp file
                temp_path = Path(f"temp_{name}.png")
                with open(temp_path, 'wb') as f:
                    f.write(buf.read())
                
                mlflow.log_artifact(str(temp_path))
                temp_path.unlink()
                
            except Exception as e:
                logger.warning(f"mlflow figure logging failed: {e}")
    
    def log_regime_statistics(
        self,
        regime_probs: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log regime assignment statistics.
        
        Args:
            regime_probs: Regime probabilities (N, K)
            step: Step number
        """
        # Compute statistics
        dominant = np.argmax(regime_probs, axis=1)
        entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10), axis=1)
        
        metrics = {
            "regime/mean_entropy": float(np.mean(entropy)),
            "regime/max_prob": float(np.max(regime_probs)),
            "regime/min_prob": float(np.min(regime_probs)),
        }
        
        # Regime usage
        for k in range(regime_probs.shape[1]):
            usage = np.sum(dominant == k)
            metrics[f"regime/usage_{k}"] = int(usage)
            metrics[f"regime/fraction_{k}"] = float(usage / len(dominant))
        
        self.log_metrics(metrics, step=step)
    
    def finish(self):
        """Finish experiment and cleanup."""
        # Wandb
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
                logger.info("✓ Wandb run finished")
            except Exception as e:
                logger.warning(f"wandb finish failed: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
                logger.info("✓ MLflow run finished")
            except Exception as e:
                logger.warning(f"mlflow finish failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


# Convenience function
def init_tracker(
    config: Any,
    backend: str = "wandb",
    project: str = "sd-mose",
    name: Optional[str] = None,
    **kwargs,
) -> Optional[ExperimentTracker]:
    """Initialize experiment tracker with error handling.
    
    Args:
        config: Configuration object
        backend: "wandb", "mlflow", "both", or "none"
        project: Project name
        name: Run name
        **kwargs: Additional arguments for ExperimentTracker
        
    Returns:
        ExperimentTracker instance or None if disabled
    """
    if backend == "none":
        logger.info("Experiment tracking disabled")
        return None
    
    try:
        tracker = ExperimentTracker(
            backend=backend,
            project=project,
            name=name,
            config=config,
            **kwargs,
        )
        return tracker
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        logger.info("Continuing without experiment tracking")
        return None
