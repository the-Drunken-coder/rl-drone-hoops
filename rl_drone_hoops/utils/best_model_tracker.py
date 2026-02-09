"""Best model tracking and saving across training runs."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BestModelTracker:
    """Track and save the best performing model across all runs.

    Maintains a JSON metadata file with the best model info and automatically
    syncs the best checkpoint to a models/ directory for cross-platform sharing.
    """

    def __init__(self, models_dir: str | Path = "models"):
        """Initialize tracker.

        Args:
            models_dir: Directory to save best models (should be git-committed)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "best_model.json"
        self.best_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load best model metadata from JSON."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load metadata: {e}, starting fresh")

        return {
            "best_return": -float("inf"),
            "best_gates": 0,
            "best_success_rate": 0.0,
            "checkpoint_path": None,
            "run_name": None,
            "step": 0,
        }

    def _save_metadata(self) -> None:
        """Save best model metadata to JSON."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.best_metadata, f, indent=2)
        logger.info(f"Saved metadata to {self.metadata_file}")

    def check_and_save(
        self,
        eval_metrics: Dict[str, float],
        checkpoint_path: str | Path,
        run_name: str,
        step: int,
    ) -> bool:
        """Check if current checkpoint is better than best. If so, save it.

        Args:
            eval_metrics: Dict with keys: eval/return_mean, eval/gates_mean, eval/finished_rate
            checkpoint_path: Path to current checkpoint file
            run_name: Name of current training run
            step: Global training step

        Returns:
            True if this checkpoint became the new best
        """
        current_return = eval_metrics.get("eval/return_mean", -float("inf"))
        current_gates = eval_metrics.get("eval/gates_mean", 0.0)
        current_success = eval_metrics.get("eval/finished_rate", 0.0)

        # Prioritize by: gates passed > success rate > return
        is_better = (
            current_gates > self.best_metadata["best_gates"]
            or (
                current_gates == self.best_metadata["best_gates"]
                and current_success > self.best_metadata["best_success_rate"]
            )
            or (
                current_gates == self.best_metadata["best_gates"]
                and current_success == self.best_metadata["best_success_rate"]
                and current_return > self.best_metadata["best_return"]
            )
        )

        if is_better:
            return self._save_best_model(
                checkpoint_path,
                run_name,
                step,
                current_return,
                current_gates,
                current_success,
            )

        return False

    def _save_best_model(
        self,
        checkpoint_path: str | Path,
        run_name: str,
        step: int,
        current_return: float,
        current_gates: float,
        current_success: float,
    ) -> bool:
        """Save checkpoint as new best model.

        Args:
            checkpoint_path: Source checkpoint path
            run_name: Name of training run
            step: Global step
            current_return: Return value
            current_gates: Gates passed
            current_success: Success rate

        Returns:
            True if save was successful
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False

            # Save with run_name and step in filename
            best_model_path = self.models_dir / f"best_model_step{step:09d}.pt"
            shutil.copy2(checkpoint_path, best_model_path)

            # Update metadata
            self.best_metadata.update(
                {
                    "best_return": current_return,
                    "best_gates": current_gates,
                    "best_success_rate": current_success,
                    "checkpoint_path": str(best_model_path),
                    "run_name": run_name,
                    "step": step,
                }
            )
            self._save_metadata()

            logger.info(
                f"New best model! gates={current_gates:.1f}, "
                f"success={current_success:.1%}, return={current_return:.1f} "
                f"at step {step} from run {run_name}"
            )
            print(
                f"\n*** NEW BEST MODEL ***\n"
                f"  Gates: {current_gates:.1f}\n"
                f"  Success rate: {current_success:.1%}\n"
                f"  Return: {current_return:.1f}\n"
                f"  Saved to: {best_model_path}\n",
                flush=True,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            return False

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best saved model."""
        path = self.best_metadata.get("checkpoint_path")
        if path and Path(path).exists():
            return Path(path)
        return None

    def get_best_metrics(self) -> Dict[str, Any]:
        """Get metrics of best model."""
        return {
            "return": self.best_metadata["best_return"],
            "gates": self.best_metadata["best_gates"],
            "success_rate": self.best_metadata["best_success_rate"],
            "step": self.best_metadata["step"],
            "run_name": self.best_metadata["run_name"],
        }
