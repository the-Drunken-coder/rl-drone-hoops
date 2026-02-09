"""Tests for JAX ↔ PyTorch bridge utilities."""
import numpy as np
import pytest

from rl_drone_hoops.utils.jax_torch_bridge import jax_to_numpy

# Conditionally import JAX/torch – tests are skipped if unavailable.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestJaxToNumpy:
    """Tests for jax_to_numpy conversion."""

    def test_float32(self):
        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        result = jax_to_numpy(a)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_uint8(self):
        a = jnp.array([0, 128, 255], dtype=jnp.uint8)
        result = jax_to_numpy(a)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 128, 255])

    def test_multidimensional(self):
        a = jnp.ones((4, 3, 2), dtype=jnp.float32)
        result = jax_to_numpy(a)
        assert result.shape == (4, 3, 2)

    def test_scalar(self):
        a = jnp.float32(3.14)
        result = jax_to_numpy(a)
        assert result.shape == ()
        np.testing.assert_almost_equal(result, 3.14, decimal=5)


class TestJaxToTorch:
    """Tests for jax_to_torch conversion (requires PyTorch)."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def test_float32_cpu(self):
        import torch
        from rl_drone_hoops.utils.jax_torch_bridge import jax_to_torch

        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        t = jax_to_torch(a, device="cpu")
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])

    def test_uint8_cpu(self):
        from rl_drone_hoops.utils.jax_torch_bridge import jax_to_torch

        a = jnp.array([0, 128, 255], dtype=jnp.uint8)
        t = jax_to_torch(a, device="cpu")
        assert t.dtype.is_floating_point is False
        np.testing.assert_array_equal(t.numpy(), [0, 128, 255])

    def test_shape_preserved(self):
        from rl_drone_hoops.utils.jax_torch_bridge import jax_to_torch

        a = jnp.ones((2, 3, 4), dtype=jnp.float32)
        t = jax_to_torch(a, device="cpu")
        assert t.shape == (2, 3, 4)


class TestTorchToJax:
    """Tests for torch_to_jax conversion (requires PyTorch)."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def test_float32(self):
        import torch
        from rl_drone_hoops.utils.jax_torch_bridge import torch_to_jax

        t = torch.tensor([1.0, 2.0, 3.0])
        a = torch_to_jax(t)
        np.testing.assert_array_equal(np.asarray(a), [1.0, 2.0, 3.0])

    def test_roundtrip(self):
        import torch
        from rl_drone_hoops.utils.jax_torch_bridge import jax_to_torch, torch_to_jax

        original = jnp.array([1.5, -2.5, 0.0], dtype=jnp.float32)
        roundtripped = torch_to_jax(jax_to_torch(original, device="cpu"))
        np.testing.assert_allclose(
            np.asarray(roundtripped), np.asarray(original), atol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
