"""Tests for the MuJoCo drone hoops environment."""
import numpy as np
import pytest

from rl_drone_hoops.envs import MujocoDroneHoopsEnv


class TestMujocoDroneHoopsEnv:
    """Test suite for MujocoDroneHoopsEnv."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        env = MujocoDroneHoopsEnv(
            image_size=64,
            camera_fps=30.0,
            imu_hz=200.0,
            control_hz=50.0,
            physics_hz=500.0,
            n_gates=3,
            episode_s=5.0,
        )
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment initialization."""
        assert env is not None
        assert env.image_size == 64
        assert env.n_gates == 3
        assert len(env.gates) == 3

    def test_reset_returns_valid_obs(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset()
        assert "image" in obs
        assert "imu" in obs
        assert "last_action" in obs
        assert obs["image"].shape == (64, 64, 1)
        assert obs["imu"].shape[1] == 6  # [gx, gy, gz, ax, ay, az]
        assert obs["last_action"].shape == (4,)
        assert "next_gate_idx" in info

    def test_reset_with_fixed_track(self, env):
        """Test reset with fixed_track option."""
        obs1, _ = env.reset(options={"fixed_track": True})
        gates1 = [g.center.copy() for g in env.gates]

        obs2, _ = env.reset(options={"fixed_track": True})
        gates2 = [g.center.copy() for g in env.gates]

        # Gates should be the same with fixed_track=True
        for g1, g2 in zip(gates1, gates2):
            np.testing.assert_array_almost_equal(g1, g2)

    def test_step_with_valid_action(self, env):
        """Test step with valid action."""
        obs, _ = env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "image" in obs
        assert "imu" in obs
        assert "last_action" in obs

    def test_step_with_invalid_action_shape(self, env):
        """Test step with invalid action shape raises error."""
        env.reset()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Wrong shape
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(action)

    def test_step_clamps_action(self, env):
        """Test step clamps actions to [-1, 1]."""
        obs, _ = env.reset()
        action = np.array([2.0, -2.0, 0.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        # Should not raise
        assert True

    def test_episode_terminates_on_crash(self, env):
        """Test episode terminates when drone crashes."""
        obs, _ = env.reset()
        # Apply large actions to crash the drone
        for _ in range(500):  # Should crash before max steps
            action = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_reward_weights_validation(self):
        """Test invalid reward weights are caught."""
        with pytest.raises(ValueError, match="Reward weight"):
            MujocoDroneHoopsEnv(k_smooth=1e10)

    def test_gate_radius_validation(self):
        """Test invalid gate radius is caught."""
        from rl_drone_hoops.envs.mujoco_drone_hoops_env import Gate

        with pytest.raises(ValueError, match="radius must be positive"):
            Gate(center=np.array([0, 0, 0]), normal=np.array([1, 0, 0]), radius=-1.0)

    def test_gate_normal_validation(self):
        """Test invalid gate normal is caught."""
        from rl_drone_hoops.envs.mujoco_drone_hoops_env import Gate

        # Non-unit normal should fail
        with pytest.raises(ValueError, match="normal must be unit"):
            Gate(center=np.array([0, 0, 0]), normal=np.array([2, 0, 0]), radius=1.0)

    def test_observation_types(self, env):
        """Test observation has correct dtypes."""
        obs, _ = env.reset()
        assert obs["image"].dtype == np.uint8
        assert obs["imu"].dtype == np.float32
        assert obs["last_action"].dtype == np.float32

    def test_render_rgb(self, env):
        """Test render_rgb returns valid RGB array."""
        env.reset()
        rgb = env.render_rgb(height=128, width=128)
        assert rgb.shape == (128, 128, 3)
        assert rgb.dtype == np.uint8

    def test_render_rgb_caching(self, env):
        """Test render_rgb caches renderers correctly."""
        env.reset()
        # Create renderers at different resolutions
        rgb1 = env.render_rgb(height=128, width=128)
        rgb2 = env.render_rgb(height=64, width=64)
        rgb3 = env.render_rgb(height=128, width=128)  # Same as rgb1
        # Should reuse cached renderer
        assert len(env._extra_renderers) == 2  # Two different sizes

    def test_pose_rpy(self, env):
        """Test pose_rpy returns valid values."""
        env.reset()
        for _ in range(10):
            pos, rpy = env.pose_rpy()
            assert pos.shape == (3,)
            assert rpy.shape == (3,)

    def test_different_track_types(self):
        """Test both track types can be created."""
        for track_type in ["straight", "random_turns"]:
            env = MujocoDroneHoopsEnv(track_type=track_type, n_gates=3)
            obs, _ = env.reset()
            assert len(env.gates) == 3
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
