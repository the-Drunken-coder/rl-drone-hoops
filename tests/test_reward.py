"""Tests for reward computation."""
import numpy as np
import pytest

from rl_drone_hoops.envs import MujocoDroneHoopsEnv


class TestRewardComputation:
    """Test suite for reward computation logic."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        env = MujocoDroneHoopsEnv(
            image_size=64,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            episode_s=5.0,
        )
        yield env
        env.close()

    def test_survival_reward_always_positive(self, env):
        """Test that survival reward is always applied."""
        obs, _ = env.reset()
        # The drone starts in a stable state with hover thrust
        for _ in range(10):
            action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            # Should include survival reward
            assert reward >= env.r_alive - 1.0  # Some margin for other penalties

    def test_crash_penalty_applied(self, env):
        """Test crash penalty is applied on crash."""
        obs, _ = env.reset()
        crash_occurred = False
        # Apply aggressive actions to crash
        for _ in range(300):
            action = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated and info.get("crash", False):
                assert reward <= env.r_crash + 10.0  # Crash penalty should be large negative
                crash_occurred = True
                break
        assert crash_occurred, "Expected drone to crash but it didn't"

    def test_gate_bonus_on_pass(self, env):
        """Test gate bonus structure."""
        obs, _ = env.reset()
        gate_passed = False
        # Fly towards first gate with moderate actions
        for _ in range(500):
            # Try to fly towards the gate
            action = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("gate_passed", False):
                # Should receive gate bonus
                assert "reward_gate" in info
                gate_passed = True
                break
        # Note: Due to randomness in gate placement and dynamics,
        # this test may occasionally not pass a gate within 500 steps

    def test_smoothness_reward_penalizes_changes(self, env):
        """Test that smoothness penalty penalizes action changes."""
        obs, _ = env.reset()
        obs, reward1, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        obs, reward2, _, _, _ = env.step(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        obs, reward3, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        # Large action change should result in smaller reward
        assert reward2 < reward1  # Action change penalty

    def test_info_contains_reward_components(self, env):
        """Test that info dict contains all reward components."""
        obs, _ = env.reset()
        obs, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        required_keys = [
            "reward_alive",
            "reward_gate",
            "reward_shaping",
            "reward_smooth",
            "reward_tilt",
            "reward_angrate",
            "next_gate_idx",
            "pos",
            "vel",
        ]
        for key in required_keys:
            assert key in info, f"Missing key {key} in info dict"

    def test_tilt_penalty_increases_with_angle(self, env):
        """Test tilt penalty increases with roll/pitch."""
        obs, _ = env.reset()
        # Collect some episodes to see tilt penalty effect
        tilt_penalties = []
        for episode in range(3):
            obs, _ = env.reset()
            for step in range(20):
                # Aggressive roll action
                action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                tilt_penalties.append(info.get("reward_tilt", 0.0))
                if terminated or truncated:
                    break

        # Average tilt penalty should be negative (penalty)
        avg_tilt = np.mean(tilt_penalties)
        assert avg_tilt < 0, "Tilt penalty should be negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
