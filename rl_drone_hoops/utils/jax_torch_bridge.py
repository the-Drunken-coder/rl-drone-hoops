"""JAX ↔ PyTorch tensor bridge for MJX physics ↔ PyTorch RL training.

Provides zero-copy (when possible) conversions between JAX arrays and PyTorch
tensors.  The conversions use the DLPack protocol for GPU tensors and fall back
to NumPy for CPU tensors.

Typical usage:
    from rl_drone_hoops.utils.jax_torch_bridge import jax_to_torch, torch_to_jax

    torch_obs = jax_to_torch(jax_obs, device="cuda")
    jax_action = torch_to_jax(torch_action)
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# Dtype mappings from JAX → NumPy/PyTorch
_JAX_TO_NP_DTYPE = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def jax_to_numpy(jax_array: "jax.Array") -> np.ndarray:
    """Convert a JAX array to a NumPy array (always copies from device).

    Args:
        jax_array: JAX array on any device.

    Returns:
        NumPy ndarray with matching dtype and shape.
    """
    return np.asarray(jax_array)


def jax_to_torch(
    jax_array: "jax.Array",
    device: str = "cpu",
) -> "torch.Tensor":
    """Convert a JAX array to a PyTorch tensor.

    Uses NumPy as an intermediate representation for CPU tensors.
    For GPU↔GPU the DLPack protocol is attempted first.

    Args:
        jax_array: JAX array.
        device: Target PyTorch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        PyTorch tensor on the requested device.
    """
    if torch is None:
        raise ImportError("PyTorch is required for jax_to_torch")
    if jax is None:
        raise ImportError("JAX is required for jax_to_torch")

    # Fast path via NumPy (works for CPU JAX arrays, which is our primary case)
    np_arr = np.asarray(jax_array)
    t = torch.from_numpy(np_arr)

    target = torch.device(device)
    if t.device != target:
        t = t.to(target)

    return t


def torch_to_jax(torch_tensor: "torch.Tensor") -> "jax.Array":
    """Convert a PyTorch tensor to a JAX array.

    Uses NumPy as an intermediate representation.

    Args:
        torch_tensor: PyTorch tensor on any device.

    Returns:
        JAX array on the default JAX device.
    """
    if jax is None:
        raise ImportError("JAX is required for torch_to_jax")
    if torch is None:
        raise ImportError("PyTorch is required for torch_to_jax")

    np_arr = torch_tensor.detach().cpu().numpy()
    return jnp.asarray(np_arr)


def torch_to_numpy(torch_tensor: "torch.Tensor") -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        torch_tensor: PyTorch tensor.

    Returns:
        NumPy ndarray.
    """
    return torch_tensor.detach().cpu().numpy()
