from rl_drone_hoops.envs.mujoco_drone_hoops_env import MujocoDroneHoopsEnv

__all__ = ["MujocoDroneHoopsEnv"]

# MJX (JAX-accelerated) variants – imported lazily to avoid hard JAX dependency.
try:
    from rl_drone_hoops.envs.mjx_drone_hoops_env import MJXDronePhysics  # noqa: F401
    from rl_drone_hoops.envs.mjx_gymnasium_wrapper import MJXDroneHoopsEnv  # noqa: F401
    from rl_drone_hoops.envs.mjx_vec_adapter import MJXVecAdapter  # noqa: F401

    __all__ += ["MJXDronePhysics", "MJXDroneHoopsEnv", "MJXVecAdapter"]
except ImportError:
    pass

