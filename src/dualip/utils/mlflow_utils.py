import getpass
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Union

import mlflow
import torch

from dualip.types import ObjectiveResult


@dataclass
class MLflowConfig:
    """Configuration for MLflow logging."""

    enabled: bool
    tracking_uri: str = ""
    experiment_name: str = ""
    run_name: str = ""
    log_hyperparameters: bool = True
    log_metrics: bool = True
    synchronous: bool = False


# Global MLflow state
class MLflowState:
    """Global state for MLflow logging."""

    def __init__(self):
        self.config: Optional[MLflowConfig] = None
        self.active_run = None
        self._enabled = False

    def set_config(self, config: MLflowConfig):
        """Set the MLflow configuration."""
        self.config = config
        self._enabled = config.enabled and is_mlflow_available()

    def is_enabled(self) -> bool:
        """Check if MLflow logging is enabled."""
        return self._enabled and self.config is not None

    def get_config(self) -> Optional[MLflowConfig]:
        """Get the current MLflow configuration."""
        return self.config

    def set_active_run(self, run):
        """Set the active MLflow run."""
        self.active_run = run

    def get_active_run(self):
        """Get the active MLflow run."""
        return self.active_run


@contextmanager
def mlflow_run_context(config: MLflowConfig):
    """
    Context manager for MLflow runs that handles setup and cleanup safely.
    If MLflow is not available or config.enabled is False, it gracefully skips logging.
    """
    # Set global state
    _mlflow_state.set_config(config)

    if not config.enabled or not is_mlflow_available():
        yield None
        return

    try:
        # Set tracking URI if provided
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        # Set experiment
        experiment_name = f"{config.experiment_name if config.experiment_name else 'dualip_experiments'}"

        experiment = mlflow.set_experiment(experiment_name)
        print(f"Created MLflow experiment name: {experiment_name} id: {experiment.experiment_id}")
        # Start run
        run_name = config.run_name if config.run_name else "dualip_run"

        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
            _mlflow_state.set_active_run(run)
            print(f"Started MLflow run: {run_name} id: {run.info.run_id}")
            yield run
    except Exception as e:
        print(f"MLflow logging failed: {e}. Continuing without MLflow logging.")
        yield None
    finally:
        # Clean up global state
        _mlflow_state.set_config(MLflowConfig(enabled=False))
        _mlflow_state.set_active_run(None)


def log_hyperparameters(params: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log hyperparameters to MLflow with selective logging based on configuration.
    """
    if not _mlflow_state.is_enabled():
        return

    config = _mlflow_state.get_config()
    if not config.log_hyperparameters:
        return

    try:

        for key, value in params.items():
            if key == "solver":
                _log_solver_params(value)
            elif key == "objective":
                _log_objective_params(value)
    except Exception as e:
        print(f"Failed to log hyperparameters: {e}")


def _log_solver_params(solver_params: Dict[str, Any], params_to_log: Optional[Set[str]] = None) -> None:
    """Log solver parameters selectively."""
    if params_to_log is None:
        # Default important solver parameters
        params_to_log = {"max_iter", "initial_step_size", "max_step_size", "gamma", "gamma_decay_type"}

    for key, value in solver_params.items():
        if key in params_to_log:
            _log_single_param(f"solver.{key}", value)


def _log_objective_params(objective_params: Dict[str, Any], params_to_log: Optional[Set[str]] = None) -> None:
    """Log objective parameters selectively."""
    if params_to_log is None:
        # Default important objective parameters
        params_to_log = {"objective_type"}

    for key, value in objective_params.items():
        if key in params_to_log:
            _log_single_param(f"objective.{key}", value)


def _log_single_param(key: str, value: Any) -> None:
    """Log a single parameter, handling type conversion."""
    try:
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                mlflow.log_param(key, value.item())
        else:
            mlflow.log_param(key, str(value))
    except Exception as e:
        print(f"Failed to log parameter {key}: {e}")


def log_metrics(metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow. Handles various numeric types and tensors.
    """
    if not _mlflow_state.is_enabled():
        return

    config = _mlflow_state.get_config()
    if not config.log_metrics:
        return

    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, str)):
                if step is not None:
                    mlflow.log_metric(key, value, step=step, synchronous=config.synchronous)
                else:
                    mlflow.log_metric(key, value, synchronous=config.synchronous)
            else:
                print(f"Skipped metric {key} (type: {type(value).__name__})")
    except Exception as e:
        print(f"Failed to log metrics: {e}")


def log_objective_result(result: ObjectiveResult, step: Optional[int] = None) -> None:
    """
    Log objective function results to MLflow.
    """
    if not _mlflow_state.is_enabled():
        return

    try:
        metrics = {}

        if result.dual_objective is not None:
            metrics["dual_objective"] = result.dual_objective.item()
        if result.primal_objective is not None:
            metrics["primal_objective"] = result.primal_objective.item()

        if result.reg_penalty is not None:
            metrics["regularization_penalty"] = result.reg_penalty.item()

        if result.max_pos_slack is not None:
            metrics["max_positive_slack"] = result.max_pos_slack.item()

        if result.sum_pos_slack is not None:
            metrics["sum_positive_slack"] = result.sum_pos_slack.item()

        if metrics:
            log_metrics(metrics, step)
    except Exception as e:
        print(f"Failed to log objective result: {e}")


def is_mlflow_available() -> bool:
    """Check if MLflow is available and properly configured."""
    try:
        import mlflow

        return True
    except ImportError:
        return False


# Global instance
_mlflow_state = MLflowState()
