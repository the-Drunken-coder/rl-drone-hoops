# Isaac Gym Environment Rewrite

## Overview

This document describes the NVIDIA Isaac Gym integration for the RL Drone Hoops
environment. The Isaac Gym backend enables 100s to 1000s of parallel
GPU-accelerated physics simulations, dramatically improving training throughput.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              IsaacDroneHoopsEnv (Gymnasium)             │
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────┐  │
│  │ IsaacDrone-   │ │ IsaacSensors  │ │ Reward &      │  │
│  │ Physics       │ │ (Camera+IMU)  │ │ Termination   │  │
│  └──────┬───────┘ └───────┬───────┘ └───────┬───────┘  │
│         │                 │                 │           │
│  ┌──────┴─────────────────┴─────────────────┴───────┐  │
│  │         IsaacGymBase (simulation core)            │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │   Isaac Gym SDK / PhysX  (GPU)   ──OR──   CPU     │  │
│  │                                       fallback    │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### Module Summary

| Module | File | Purpose |
|--------|------|---------|
| Base environment | `envs/isaac_gym_env.py` | Isaac Gym simulation setup and lifecycle |
| Drone physics | `envs/isaac_drone_physics.py` | Vectorized drone dynamics (forces, torques, integration) |
| Sensors | `envs/isaac_sensors.py` | Batched camera and IMU with latency simulation |
| Gymnasium wrapper | `envs/isaac_gymnasium_env.py` | Standard RL interface (`reset`, `step`, `close`) |
| Assets | `assets/isaac_gym_assets.py` | URDF generation for drone and gates, track layout |

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- PyTorch 2.0+ with CUDA

### Isaac Gym SDK (Optional)

Isaac Gym requires registration with NVIDIA:

1. Visit <https://developer.nvidia.com/isaac-gym>
2. Download the Isaac Gym Preview release
3. Install following the included instructions:
   ```bash
   cd isaacgym/python
   pip install -e .
   ```

**Note:** The environment works without Isaac Gym installed by using a CPU
tensor-based physics fallback. This is suitable for development and testing but
does not provide GPU acceleration.

### Quick Start

```bash
# Install base dependencies
pip install -r requirements.txt

# Install training dependencies (includes PyTorch)
pip install -r requirements-train.txt

# Run tests (no Isaac Gym required)
pytest tests/test_isaac_physics.py tests/test_isaac_env.py -v
```

## Usage

### Python API

```python
from rl_drone_hoops.envs.isaac_gymnasium_env import IsaacDroneHoopsEnv, IsaacEnvConfig

# Create environment with 256 parallel simulations
config = IsaacEnvConfig(
    num_envs=256,
    image_size=96,
    n_gates=3,
    device="cuda:0",  # or "cpu" for fallback
)
env = IsaacDroneHoopsEnv(config=config)

# Standard Gymnasium interface
obs, info = env.reset(seed=42)

for _ in range(1000):
    actions = torch.randn(256, 4).clamp(-1, 1)  # (num_envs, 4)
    obs, reward, terminated, truncated, info = env.step(actions)

env.close()
```

### Training with Isaac Gym

```bash
# Train with Isaac Gym (falls back to CPU tensors if SDK not installed)
python3 scripts/train_recurrent_ppo.py --use-isaac --num-envs 256

# Train with MuJoCo (original behavior)
python3 scripts/train_recurrent_ppo.py --num-envs 4
```

### Configuration

Isaac Gym settings are in `config/default.yaml`:

```yaml
isaac:
  enabled: false              # Use Isaac Gym instead of MuJoCo
  num_envs: 256               # Number of parallel GPU environments
  spacing: 10.0               # Spacing between environments (meters)
  substeps: 2                 # Physics substeps per simulation step
  use_gpu_pipeline: true      # Use GPU pipeline for tensor access
  compute_device_id: 0        # CUDA device for compute
  graphics_device_id: -1      # CUDA device for rendering (-1 = headless)
```

## Physics

### Drone Model

The quadrotor drone is modeled with:

- **Mass:** 0.5 kg
- **Inertia:** Ixx=0.0023, Iyy=0.0023, Izz=0.004 kg·m²
- **Max thrust:** 12 N total (4 × 3 N per motor)
- **Max body rate:** 8 rad/s
- **Arm length:** 0.15 m (X-configuration)

### Actuator Dynamics

First-order lag filters model motor response:

- **Thrust lag:** τ = 20 ms
- **Rate lag:** τ = 10 ms

### Forces

- **Gravity:** -9.81 m/s² along Z
- **Thrust:** Along body Z-axis
- **Drag:** Linear (0.1) + quadratic (0.01)

### Integration

Semi-implicit Euler with quaternion normalization at each step.

### Physics Calibration

PhysX (Isaac Gym) and MuJoCo have different internal dynamics. Expect ±10%
variation in trajectory metrics. Key tuning parameters:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `mass` | `DronePhysicsConfig` | Overall inertia |
| `max_thrust` | `DronePhysicsConfig` | Hover/climb capability |
| `thrust_tau` | `DronePhysicsConfig` | Motor response speed |
| `drag_coeff_*` | `DronePhysicsConfig` | Terminal velocity |

## Sensors

### Camera (FPV)

- 96×96 grayscale (configurable)
- 60 Hz default (configurable)
- 20 ms simulated latency
- In CPU fallback mode: synthetic gradient image

### IMU

- 6-DOF: 3-axis gyroscope + 3-axis accelerometer
- 400 Hz default (configurable)
- 2 ms simulated latency
- Windowed history (last N samples)
- Optional noise injection (disabled by default)

## Rewards

Vectorized reward computation matching the MuJoCo environment:

| Component | Value | Description |
|-----------|-------|-------------|
| Gate passed | +10.0 | Per gate successfully traversed |
| Survival | +0.1 | Per timestep alive |
| Progress | ±1.0 | Distance change toward next gate |
| Centering | 0–0.5 | Proximity to gate center axis |
| Smoothness | -0.05 | Penalty for jerky actions |
| Crash | -20.0 | On ground collision or excessive tilt |

## Termination Conditions

- **Crash:** Ground contact (z ≤ 0.01) or tilt > 75°
- **Out of bounds:** |x| > 50m, |y| > 50m, or z > 20m
- **Truncation:** Episode exceeds maximum duration

## Performance Targets

| Setup | Envs | Expected Steps/sec |
|-------|------|--------------------|
| MuJoCo (CPU, 4 envs) | 4 | ~200 |
| Isaac Gym (RTX 3060, 256 envs) | 256 | ~5,000 |
| Isaac Gym (RTX 4090, 1000 envs) | 1000 | ~20,000+ |

## Troubleshooting

### Isaac Gym not found

The environment falls back to CPU tensor-based physics automatically.
Install Isaac Gym SDK for GPU acceleration.

### CUDA out of memory

Reduce `num_envs` in the configuration. Start with 256 and increase
based on available GPU memory.

### Physics instability

If the drone jitters or explodes:
1. Reduce `physics_hz` (try 500 Hz)
2. Increase `substeps` (try 4)
3. Reduce `max_thrust` or `max_rate`
4. Check quaternion normalization in logs

### Dynamics mismatch with MuJoCo

PhysX and MuJoCo produce different results. To calibrate:
1. Run both environments with identical actions
2. Compare position/velocity trajectories
3. Adjust `DronePhysicsConfig` parameters
4. Target ±5% on key metrics

## File Structure

```
rl_drone_hoops/
├── assets/
│   ├── __init__.py
│   └── isaac_gym_assets.py        # URDF generation, track layout
├── envs/
│   ├── isaac_gym_env.py           # Base Isaac Gym simulation
│   ├── isaac_drone_physics.py     # Vectorized drone dynamics
│   ├── isaac_sensors.py           # Camera + IMU pipeline
│   └── isaac_gymnasium_env.py     # Gymnasium wrapper
tests/
├── test_isaac_physics.py          # Physics + quaternion tests
└── test_isaac_env.py              # Environment integration tests
docs/
└── ISAAC_GYM_REWRITE.md           # This document
```
