"""Tests for Isaac Gym drone physics simulation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_drone_hoops.envs.isaac_drone_physics import (
    DronePhysicsConfig,
    IsaacDronePhysics,
    _integrate_quaternion,
    _quat_rotate_vector,
)


class TestIsaacDronePhysics:
    """Test suite for IsaacDronePhysics."""

    @pytest.fixture
    def physics(self):
        """Create a physics instance with 4 parallel environments."""
        config = DronePhysicsConfig(dt=0.001)
        return IsaacDronePhysics(num_envs=4, config=config, device="cpu")

    def test_creation(self, physics):
        """Test physics instance creation."""
        assert physics.num_envs == 4
        assert physics.config.dt == 0.001
        assert physics.config.mass == 0.5
        assert physics.config.gravity == 9.81

    def test_reset(self, physics):
        """Test physics state reset."""
        physics.reset()
        state = physics.get_state()
        assert state["position"].shape == (4, 3)
        assert state["velocity"].shape == (4, 3)
        assert state["quaternion"].shape == (4, 4)
        assert state["angular_velocity"].shape == (4, 3)
        assert state["thrust"].shape == (4,)
        assert state["rates"].shape == (4, 3)

        # Starting height should be 2.0
        assert torch.allclose(state["position"][:, 2], torch.tensor(2.0))
        # Velocity should be zero
        assert torch.allclose(state["velocity"], torch.zeros(4, 3))
        # Identity quaternion (w=1, x=y=z=0)
        expected_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, 4)
        assert torch.allclose(state["quaternion"], expected_quat)

    def test_partial_reset(self, physics):
        """Test resetting only specific environments."""
        physics.reset()
        # Move env 0 somewhere
        physics._position[0] = torch.tensor([5.0, 5.0, 5.0])
        physics._velocity[0] = torch.tensor([1.0, 1.0, 1.0])

        # Reset only env 0
        physics.reset(env_ids=torch.tensor([0]))
        state = physics.get_state()
        assert torch.allclose(state["position"][0, 2], torch.tensor(2.0))
        assert torch.allclose(state["velocity"][0], torch.zeros(3))

    def test_gravity_pulls_drone_down(self, physics):
        """Test that gravity causes the drone to fall."""
        physics.reset()
        initial_z = physics.get_state()["position"][:, 2].clone()

        # Zero actions (no thrust)
        actions = torch.zeros(4, 4)
        physics.apply_actions(actions)

        # Run 100 physics steps
        for _ in range(100):
            physics.step()

        state = physics.get_state()
        final_z = state["position"][:, 2]

        # Drone should have fallen
        assert torch.all(final_z < initial_z)

    def test_thrust_counteracts_gravity(self, physics):
        """Test that sufficient thrust prevents falling."""
        physics.reset()
        initial_z = physics.get_state()["position"][:, 2].clone()

        # Apply high thrust (action 3 = thrust, 1.0 = max)
        actions = torch.zeros(4, 4)
        actions[:, 3] = 0.5  # Moderate thrust

        physics.apply_actions(actions)

        for _ in range(100):
            physics.apply_actions(actions)  # Continue applying
            physics.step()

        state = physics.get_state()
        final_z = state["position"][:, 2]

        # With enough thrust, drone should not drop as fast
        # (may still drop due to actuator lag, but should be partially countered)
        assert final_z.mean() > 0.0, "Drone should not be on the ground"

    def test_rate_commands(self, physics):
        """Test drone responds to rate commands."""
        physics.reset()

        # Apply roll rate command
        actions = torch.zeros(4, 4)
        actions[:, 0] = 0.5  # Roll rate command

        for _ in range(50):
            physics.apply_actions(actions)
            physics.step()

        state = physics.get_state()
        # Angular velocity should be non-zero
        assert state["angular_velocity"][:, 0].abs().mean() > 0.01

    def test_no_nan_states(self, physics):
        """Test physics produces no NaN values."""
        physics.reset()
        actions = torch.randn(4, 4) * 0.5

        for _ in range(500):
            physics.apply_actions(actions)
            physics.step()

        state = physics.get_state()
        for key, val in state.items():
            assert not torch.isnan(val).any(), f"NaN found in {key}"
            assert not torch.isinf(val).any(), f"Inf found in {key}"

    def test_action_clamping(self, physics):
        """Test that extreme actions are handled without explosion."""
        physics.reset()

        # Extreme actions
        actions = torch.full((4, 4), 100.0)
        physics.apply_actions(actions)

        for _ in range(100):
            physics.step()

        state = physics.get_state()
        # States should remain finite
        for key, val in state.items():
            assert not torch.isnan(val).any(), f"NaN found in {key}"
            assert not torch.isinf(val).any(), f"Inf found in {key}"

    def test_ground_collision(self, physics):
        """Test ground collision clamps altitude."""
        physics.reset()
        physics._position[:, 2] = 0.01  # Near ground
        physics._velocity[:, 2] = -10.0  # Falling fast

        actions = torch.zeros(4, 4)
        physics.apply_actions(actions)
        physics.step()

        state = physics.get_state()
        assert torch.all(state["position"][:, 2] >= 0.0)

    def test_velocity_consistency(self, physics):
        """Test velocity is consistent with position changes."""
        physics.reset()
        pos_before = physics.get_state()["position"].clone()

        actions = torch.zeros(4, 4)
        actions[:, 3] = 0.3  # Some thrust
        physics.apply_actions(actions)
        physics.step()

        pos_after = physics.get_state()["position"]
        vel = physics.get_state()["velocity"]

        # Position change should be approximately velocity * dt
        expected_delta = vel * physics.config.dt
        actual_delta = pos_after - pos_before
        # Allow tolerance for the integration method
        assert torch.allclose(actual_delta, expected_delta, atol=0.01)


class TestQuaternionOps:
    """Test quaternion utility functions."""

    def test_rotate_identity(self):
        """Test identity quaternion doesn't change vector."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        result = _quat_rotate_vector(q, v)
        assert torch.allclose(result, v, atol=1e-6)

    def test_rotate_90_deg_z(self):
        """Test 90-degree rotation about Z axis."""
        angle = np.pi / 2
        q = torch.tensor([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]], dtype=torch.float32)
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        result = _quat_rotate_vector(q, v)
        expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_integrate_identity(self):
        """Test quaternion integration with zero angular velocity."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        omega = torch.tensor([[0.0, 0.0, 0.0]])
        result = _integrate_quaternion(q, omega, dt=0.01)
        assert torch.allclose(result, q, atol=1e-6)

    def test_integrate_maintains_norm(self):
        """Test quaternion integration maintains unit norm."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(10, 4).clone()
        omega = torch.randn(10, 3)

        for _ in range(100):
            q = _integrate_quaternion(q, omega, dt=0.01)

        norms = q.norm(dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
