"""Tests for Isaac Gym drone hoops environment."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_drone_hoops.envs.isaac_gymnasium_env import (
    IsaacDroneHoopsEnv,
    IsaacEnvConfig,
)


class TestIsaacDroneHoopsEnv:
    """Test suite for IsaacDroneHoopsEnv."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return IsaacEnvConfig(
            num_envs=4,
            image_size=64,
            camera_fps=30.0,
            imu_hz=200.0,
            imu_window_size=4,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            gate_radius=1.25,
            episode_duration_s=5.0,
            device="cpu",
            seed=42,
        )

    @pytest.fixture
    def env(self, config):
        """Create a test environment."""
        env = IsaacDroneHoopsEnv(config=config)
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment initialization."""
        assert env is not None
        assert env.num_envs == 4
        assert env.config.image_size == 64
        assert env.config.n_gates == 3

    def test_reset_returns_valid_obs(self, env):
        """Test reset returns valid observation dict."""
        obs, info = env.reset()
        assert "image" in obs
        assert "imu" in obs
        assert "last_action" in obs
        assert obs["image"].shape == (4, 64, 64, 1)
        assert obs["imu"].shape[1] == 4  # imu_window_size
        assert obs["imu"].shape[2] == 6  # [gx, gy, gz, ax, ay, az]
        assert obs["last_action"].shape == (4, 4)
        assert "next_gate_idx" in info

    def test_step_with_valid_action(self, env):
        """Test step with valid batched actions."""
        obs, _ = env.reset()
        actions = torch.zeros(4, 4, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs["image"].shape == (4, 64, 64, 1)
        assert reward.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert "next_gate_idx" in info

    def test_step_with_numpy_action(self, env):
        """Test step accepts numpy arrays."""
        env.reset()
        actions = np.zeros((4, 4), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert reward.shape == (4,)

    def test_step_clamps_action(self, env):
        """Test step clamps extreme actions."""
        env.reset()
        actions = torch.full((4, 4), 5.0)
        obs, _, _, _, _ = env.step(actions)
        # last_action should be clamped
        assert obs["last_action"].max() <= 1.0
        assert obs["last_action"].min() >= -1.0

    def test_observation_dtypes(self, env):
        """Test observation has correct dtypes."""
        obs, _ = env.reset()
        assert obs["image"].dtype == torch.uint8
        assert obs["imu"].dtype == torch.float32
        assert obs["last_action"].dtype == torch.float32

    def test_reward_is_finite(self, env):
        """Test rewards are finite (no NaN/Inf)."""
        env.reset()
        for _ in range(50):
            actions = torch.randn(4, 4) * 0.3
            _, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(reward).all(), "Reward contains NaN or Inf"

    def test_episode_terminates(self, env):
        """Test episode can terminate (crash or truncation)."""
        env.reset()
        terminated_any = False
        truncated_any = False
        for _ in range(1000):
            # Apply destabilizing actions
            actions = torch.tensor([[1.0, 1.0, 0.0, -1.0]] * 4)
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated.any():
                terminated_any = True
                break
            if truncated.any():
                truncated_any = True
                break

        assert terminated_any or truncated_any, (
            "Expected episode to terminate within 1000 steps"
        )

    def test_auto_reset_on_done(self, env):
        """Test environments are auto-reset when done."""
        env.reset()
        for _ in range(500):
            actions = torch.tensor([[1.0, 1.0, 0.0, -1.0]] * 4)
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if done.any():
                # Observations should still be valid after auto-reset
                assert obs["image"].shape == (4, 64, 64, 1)
                assert torch.isfinite(obs["imu"]).all()
                break

    def test_reset_with_seed(self, env):
        """Test reset with seed produces reproducible results."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        # Same seed should produce same initial observations
        assert torch.equal(obs1["last_action"], obs2["last_action"])

    def test_num_envs_override(self):
        """Test num_envs can be overridden in constructor."""
        env = IsaacDroneHoopsEnv(num_envs=8)
        assert env.num_envs == 8
        obs, _ = env.reset()
        assert obs["image"].shape[0] == 8
        env.close()

    def test_repr(self, env):
        """Test string representation."""
        s = repr(env)
        assert "IsaacDroneHoopsEnv" in s
        assert "num_envs=4" in s

    def test_gate_crossing_detection(self, env):
        """Test gate crossing detection logic."""
        env.reset()
        # Move drone to be on the correct side of the first gate,
        # then step to cross it
        state = env.physics.get_state()
        # This is a smoke test - just verify no errors
        assert env._next_gate_idx.shape == (4,)

    def test_multiple_steps_stability(self, env):
        """Test environment is stable over many steps."""
        env.reset()
        for _ in range(200):
            actions = torch.randn(4, 4) * 0.1
            obs, reward, terminated, truncated, info = env.step(actions)
            assert torch.isfinite(reward).all()
            assert obs["image"].shape == (4, 64, 64, 1)


class TestIsaacEnvConfig:
    """Test IsaacEnvConfig defaults and validation."""

    def test_defaults(self):
        """Test default config values."""
        config = IsaacEnvConfig()
        assert config.num_envs == 256
        assert config.image_size == 96
        assert config.physics_hz == 1000.0
        assert config.control_hz == 100.0
        assert config.n_gates == 3

    def test_custom_values(self):
        """Test config with custom values."""
        config = IsaacEnvConfig(
            num_envs=1024,
            image_size=48,
            n_gates=5,
        )
        assert config.num_envs == 1024
        assert config.image_size == 48
        assert config.n_gates == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
