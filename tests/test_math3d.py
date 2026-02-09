"""Tests for 3D math utilities."""
import numpy as np
import pytest

from rl_drone_hoops.utils.math3d import quat_to_mat, mat_to_rpy, unit, quat_from_two_vectors


class TestMath3D:
    """Test suite for 3D math utilities."""

    def test_quat_to_mat_identity(self):
        """Test identity quaternion produces identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        R = quat_to_mat(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_quat_to_mat_90_degree_rotation(self):
        """Test 90 degree rotation quaternion."""
        # Rotate 90 degrees around Z axis
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        R = quat_to_mat(q)
        # Check orthogonality
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)

    def test_mat_to_rpy_identity(self):
        """Test identity matrix produces zero rpy."""
        R = np.eye(3)
        rpy = mat_to_rpy(R)
        np.testing.assert_array_almost_equal(rpy, np.array([0, 0, 0]))

    def test_rpy_roundtrip(self):
        """Test quat_to_mat and mat_to_rpy roundtrip."""
        # Create a quaternion and convert to rotation matrix then to RPY
        q = np.array([0.7071, 0.7071, 0, 0])
        R = quat_to_mat(q)
        rpy = mat_to_rpy(R)
        # Should get a valid RPY
        assert rpy.shape == (3,)
        assert np.all(np.isfinite(rpy))

    def test_unit_vector(self):
        """Test unit vector normalization."""
        v = np.array([3.0, 4.0, 0.0])
        u = unit(v)
        assert np.abs(np.linalg.norm(u) - 1.0) < 1e-6

    def test_unit_zero_vector(self):
        """Test unit vector with zero input."""
        v = np.array([0.0, 0.0, 0.0])
        u = unit(v)
        np.testing.assert_array_almost_equal(u, np.array([0.0, 0.0, 0.0]))

    def test_quat_from_two_vectors_identity(self):
        """Test quaternion from same vectors is identity."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        q = quat_from_two_vectors(a, b)
        np.testing.assert_array_almost_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_quat_from_two_vectors_opposite(self):
        """Test quaternion from opposite vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        q = quat_from_two_vectors(a, b)
        # Magnitude should be 1
        assert np.abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_quat_from_two_vectors_perpendicular(self):
        """Test quaternion from perpendicular vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        q = quat_from_two_vectors(a, b)
        R = quat_to_mat(q)
        # Check that R maps a to b
        result = R @ a
        np.testing.assert_array_almost_equal(result, b, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
