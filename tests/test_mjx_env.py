"""Tests for the MJX drone hoops environment."""
import os

# MJX does not need a GL context; disable rendering to avoid EGL/OSMesa errors.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "disable"

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
mjx = pytest.importorskip("mujoco.mjx")

from rl_drone_hoops.envs.mjx_drone_hoops_env import MJXDronePhysics
from rl_drone_hoops.envs.mjx_gymnasium_wrapper import MJXDroneHoopsEnv


class TestMJXDronePhysics:
    """Test suite for the MJX batched physics engine."""

    @pytest.fixture
    def physics(self):
        """Create a test physics engine."""
        return MJXDronePhysics(
            num_envs=2,
            image_size=64,
            camera_fps=30.0,
            imu_hz=200.0,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            episode_s=5.0,
            seed=42,
        )

    def test_creation(self, physics):
        """Test physics engine initialization."""
        assert physics.num_envs == 2
        assert physics.n_gates == 3
        assert physics.n_substeps == 10

    def test_reset(self, physics):
        """Test reset returns valid state."""
        state = physics.reset(seed=42)
        assert state.mjx_data is not None
        assert state.t.shape == (2,)
        assert state.step_i.shape == (2,)
        assert state.next_gate_idx.shape == (2,)
        assert state.gate_centers.shape == (2, 3, 3)
        assert state.gate_normals.shape == (2, 3, 3)
        assert state.imu_history.shape[0] == 2
        assert state.imu_history.shape[2] == 6

    def test_step(self, physics):
        """Test stepping returns correct shapes."""
        state = physics.reset(seed=42)
        actions = jnp.zeros((2, 4), dtype=jnp.float32)
        new_state, rewards, dones, terminated, truncated, infos = physics.step(
            state, actions
        )
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert new_state.step_i.shape == (2,)
        # Step counter should have incremented
        np.testing.assert_array_equal(np.asarray(new_state.step_i), [1, 1])

    def test_step_no_nan(self, physics):
        """Test that stepping doesn't produce NaN."""
        state = physics.reset(seed=42)
        for _ in range(10):
            actions = jnp.zeros((2, 4), dtype=jnp.float32)
            state, rewards, dones, _, _, _ = physics.step(state, actions)
            assert not np.any(np.isnan(np.asarray(rewards))), "NaN in rewards"
            pos = np.asarray(state.p_prev)
            assert not np.any(np.isnan(pos)), "NaN in positions"

    def test_reward_is_finite(self, physics):
        """Test that rewards are finite."""
        state = physics.reset(seed=42)
        actions = jnp.array([[0.0, 0.0, 0.0, 0.5], [0.1, -0.1, 0.0, 0.3]])
        state, rewards, _, _, _, _ = physics.step(state, actions)
        assert np.all(np.isfinite(np.asarray(rewards)))

    def test_get_obs_numpy(self, physics):
        """Test observation extraction."""
        state = physics.reset(seed=42)
        obs = physics.get_obs_numpy(state)
        assert "imu" in obs
        assert "last_action" in obs
        assert obs["imu"].shape == (2, physics.imu_window_n, 6)
        assert obs["last_action"].shape == (2, 4)
        assert obs["imu"].dtype == np.float32
        assert obs["last_action"].dtype == np.float32

    def test_deterministic_reset(self, physics):
        """Test that reset with same seed produces identical state."""
        state1 = physics.reset(seed=123)
        state2 = physics.reset(seed=123)
        np.testing.assert_array_equal(
            np.asarray(state1.gate_centers),
            np.asarray(state2.gate_centers),
        )

    def test_different_seeds_different_tracks(self, physics):
        """Test that different seeds produce different tracks."""
        state1 = physics.reset(seed=1)
        state2 = physics.reset(seed=999)
        # Gate centers should differ
        c1 = np.asarray(state1.gate_centers)
        c2 = np.asarray(state2.gate_centers)
        assert not np.allclose(c1, c2)


class TestMJXDroneHoopsEnv:
    """Test suite for the Gymnasium wrapper."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        env = MJXDroneHoopsEnv(
            image_size=64,
            camera_fps=30.0,
            imu_hz=200.0,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            episode_s=5.0,
            seed=42,
        )
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment initialization."""
        assert env is not None
        assert env.image_size == 64
        assert env.n_gates == 3

    def test_reset_returns_valid_obs(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset()
        assert "image" in obs
        assert "imu" in obs
        assert "last_action" in obs
        assert obs["image"].shape == (64, 64, 1)
        assert obs["imu"].shape[1] == 6
        assert obs["last_action"].shape == (4,)
        assert "next_gate_idx" in info

    def test_step_with_valid_action(self, env):
        """Test step with valid action."""
        obs, _ = env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "image" in obs
        assert "imu" in obs
        assert "last_action" in obs

    def test_step_with_invalid_action_shape(self, env):
        """Test step with invalid action shape raises error."""
        env.reset()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(action)

    def test_observation_types(self, env):
        """Test observation has correct dtypes."""
        obs, _ = env.reset()
        assert obs["image"].dtype == np.uint8
        assert obs["imu"].dtype == np.float32
        assert obs["last_action"].dtype == np.float32

    def test_multiple_steps_no_error(self, env):
        """Test running multiple steps doesn't raise errors."""
        env.reset()
        for _ in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()


class TestMJXVecAdapter:
    """Test suite for the vectorized MJX adapter."""

    @pytest.fixture
    def vec(self):
        from rl_drone_hoops.envs.mjx_vec_adapter import MJXVecAdapter

        vec = MJXVecAdapter(
            num_envs=2,
            image_size=64,
            camera_fps=30.0,
            imu_hz=200.0,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            episode_s=5.0,
            seed=42,
        )
        yield vec
        vec.close()

    def test_reset(self, vec):
        """Test reset returns stacked observations."""
        obs = vec.reset()
        assert obs["image"].shape == (2, 64, 64, 1)
        assert obs["imu"].shape[1] == 2  or obs["imu"].ndim == 3
        assert obs["last_action"].shape == (2, 4)

    def test_step(self, vec):
        """Test step returns StepResult."""
        vec.reset()
        actions = np.zeros((2, 4), dtype=np.float32)
        result = vec.step(actions)
        assert result.reward.shape == (2,)
        assert result.done.shape == (2,)
        assert len(result.info) == 2
        assert result.obs["image"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
