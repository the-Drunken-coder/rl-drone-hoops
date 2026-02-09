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
            "flight": 0,
            "step": 0,
            "curriculum": {},  # Track curriculum params
        }

    def _save_metadata(self) -> None:
        """Save best model metadata to JSON."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.best_metadata, f, indent=2)
        logger.info(f"Saved metadata to {self.metadata_file}")

    def _curriculum_difficulty_score(self, curriculum: Dict[str, Any]) -> float:
        """Calculate difficulty score of curriculum.

        Higher score = harder curriculum.

        Args:
            curriculum: Dict with keys like n_gates, gate_radius, track_type, turn_max_deg

        Returns:
            Scalar difficulty score
        """
        n_gates = curriculum.get("n_gates", 3)
        gate_radius = curriculum.get("gate_radius", 1.25)
        track_type = curriculum.get("track_type", "straight")
        turn_max_deg = curriculum.get("turn_max_deg", 20.0)

        # Difficulty increases with: more gates, smaller radius, random turns, sharper turns
        score = 0.0
        score += n_gates * 10.0  # Each gate adds 10 difficulty points
        score += (2.0 - gate_radius) * 5.0  # Smaller radius = harder (1.25 = 3.75, 0.5 = 7.5)
        if track_type == "random_turns":
            score += 20.0
        score += turn_max_deg * 0.5

        return score

    def check_and_save(
        self,
        eval_metrics: Dict[str, float],
        checkpoint_path: str | Path,
        run_name: str,
        flight: int,
        curriculum: Dict[str, Any] | None = None,
    ) -> bool:
        """Check if current checkpoint is better than best. If so, save it.

        Curriculum-aware: only replaces old model if:
        - New curriculum is same/harder, OR
        - Metrics are significantly better (gates+50% on same curriculum)

        Args:
            eval_metrics: Dict with keys: eval/return_mean, eval/gates_mean, eval/finished_rate
            checkpoint_path: Path to current checkpoint file
            run_name: Name of current training run
            flight: Global flight/episode number
            curriculum: Dict with curriculum params (n_gates, gate_radius, track_type, turn_max_deg)

        Returns:
            True if this checkpoint became the new best
        """
        if curriculum is None:
            curriculum = {}

        current_return = eval_metrics.get("eval/return_mean", -float("inf"))
        current_gates = eval_metrics.get("eval/gates_mean", 0.0)
        current_success = eval_metrics.get("eval/finished_rate", 0.0)

        old_curriculum = self.best_metadata.get("curriculum", {})
        old_difficulty = self._curriculum_difficulty_score(old_curriculum)
        new_difficulty = self._curriculum_difficulty_score(curriculum)

        # Check curriculum difference
        curriculum_differs = old_curriculum != curriculum
        if curriculum_differs:
            logger.info(
                f"Curriculum changed: old difficulty={old_difficulty:.1f}, "
                f"new difficulty={new_difficulty:.1f}"
            )

        # Prioritize by: gates passed > success rate > return
        basic_is_better = (
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

        # If curriculum got easier, require significant improvement
        if new_difficulty < old_difficulty:
            # Require 50% more gates on easier curriculum to replace
            gates_improvement_needed = self.best_metadata["best_gates"] * 1.5
            is_better = current_gates > gates_improvement_needed
            if not is_better and curriculum_differs:
                logger.warning(
                    f"New model on EASIER curriculum rejected: "
                    f"gates={current_gates:.1f} (need {gates_improvement_needed:.1f}). "
                    f"Use --force-best-model to override."
                )
            return is_better and self._save_best_model(
                checkpoint_path,
                run_name,
                flight,
                current_return,
                current_gates,
                current_success,
                curriculum,
            )
        # Same or harder curriculum: use basic comparison
        else:
            if basic_is_better:
                return self._save_best_model(
                    checkpoint_path,
                    run_name,
                    flight,
                    current_return,
                    current_gates,
                    current_success,
                    curriculum,
                )

        return False

    def _save_best_model(
        self,
        checkpoint_path: str | Path,
        run_name: str,
        flight: int,
        current_return: float,
        current_gates: float,
        current_success: float,
        curriculum: Dict[str, Any] | None = None,
    ) -> bool:
        """Save checkpoint as new best model.

        Args:
            checkpoint_path: Source checkpoint path
            run_name: Name of training run
            flight: Global flight/episode number
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

            # Delete old best model to save space
            old_path = self.best_metadata.get("checkpoint_path")
            if old_path:
                old_path_obj = Path(old_path)
                if old_path_obj.exists():
                    try:
                        old_path_obj.unlink()
                        logger.info(f"Deleted old best model: {old_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete old best model: {e}")

            # Save with run_name and flight in filename
            best_model_path = self.models_dir / f"best_model_flight{flight:09d}.pt"
            shutil.copy2(checkpoint_path, best_model_path)

            # Update metadata
            self.best_metadata.update(
                {
                    "best_return": current_return,
                    "best_gates": current_gates,
                    "best_success_rate": current_success,
                    "checkpoint_path": str(best_model_path),
                    "run_name": run_name,
                    "flight": flight,
                    "curriculum": curriculum or {},
                }
            )
            self._save_metadata()

            logger.info(
                f"New best model! gates={current_gates:.1f}, "
                f"success={current_success:.1%}, return={current_return:.1f} "
                f"at flight {flight} from run {run_name}"
            )
            print(
                f"\n*** NEW BEST MODEL ***\n"
                f"  Gates: {current_gates:.1f}\n"
                f"  Success rate: {current_success:.1%}\n"
                f"  Return: {current_return:.1f}\n"
                f"  Flight: {flight}\n"
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
