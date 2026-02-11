# Training Configuration Files

This directory contains YAML configuration files for training the RL drone policy. All settings can be defined in YAML and overridden via CLI arguments.

## Available Configs

### `default.yaml`
The default training configuration with balanced settings suitable for most use cases.
- 4 parallel environments
- 200K training steps
- 50K eval frequency
- Standard hyperparameters

### `fast-train.yaml`
CPU-friendly configuration for quick testing and iteration.
- 1 environment (single-threaded)
- 4K training steps
- Reduced physics/camera rates (20-500 Hz)
- Smaller 64x64 images
- Good for laptop/development

### `performance.yaml`
High-performance configuration for powerful GPUs.
- 16 parallel environments
- 1M training steps
- 4-stage curriculum learning
- Larger 128x128 images
- 4 MuJoCo threads per env
- More frequent checkpointing

## Usage

### Use a specific config file
```bash
python3 scripts/train_recurrent_ppo.py --config config/fast-train.yaml
```

### Use default config
```bash
python3 scripts/train_recurrent_ppo.py
```

### Override config values via CLI (takes precedence over config file)
```bash
# Change eval frequency to 5000 steps
python3 scripts/train_recurrent_ppo.py --eval-every-steps 5000

# Change multiple values
python3 scripts/train_recurrent_ppo.py --num-envs 8 --lr 2e-4 --gate-radius 1.0

# Resume training with new settings
python3 scripts/train_recurrent_ppo.py --run-dir runs/my_run --resume --eval-every-steps 5000
```

## Priority Order

Settings are applied in this order (later overrides earlier):
1. **Config file** (default.yaml or specified with `--config`)
2. **Checkpoint config** (if resuming, values from the original training run)
3. **CLI arguments** (highest priority, always override everything)

This allows you to:
- Define base defaults in the config file
- Modify specific settings without rewriting the config
- Resume training and change parameters on the fly

## Configuration Sections

### `ppo` - PPO Algorithm & Training
- `seed`: Random seed
- `device`: "auto", "cpu", or "cuda"
- `num_envs`: Parallel environments (use `0` for auto: detect CPU cores)
- `total_steps`: Total environment steps
- `rollout_steps`: Steps per PPO update
- `gamma`, `gae_lambda`, `clip_coef`, etc.: PPO hyperparameters
- `lr`: Learning rate
- `adam_eps`: Adam optimizer epsilon
- `update_epochs`: Passes over rollout data
- `minibatch_envs`: Environments per minibatch

### `environment` - Sensors & Physics
- `image_size`: FPV camera resolution (square pixels)
- `image_rot90`: Rotate camera 0-3 times
- `camera_fps`: Camera sampling rate
- `camera_latency_ms`: Approximate camera latency
- `imu_hz`: IMU sampling rate
- `imu_latency_ms`: Approximate IMU latency
- `control_hz`: Control loop rate
- `control_latency_ms`: Approximate action latency
- `physics_hz`: Physics simulation rate
- `max_tilt_deg`: Max roll/pitch before crash

### `reward` - Reward Weights
- `r_alive`: Survival reward per step
- `r_gate`: Bonus for passing a gate
- `k_progress`: Reward for reducing distance to next gate
- `k_center`: Penalty for lateral distance from gate axis
- `k_speed`: Reward for velocity toward the gate
- `k_heading`: Reward for yaw alignment toward the gate (horizontal only)
- `k_away`: Penalty for increasing distance to the gate
- `k_smooth`: Penalty for action changes
- `k_tilt`: Penalty for roll/pitch magnitude
- `k_angrate`: Penalty for angular rate magnitude
- `r_crash`: Penalty applied on crash

### `track` - Course Generation
- `track_type`: "straight" or "random_turns"
- `n_gates`: Number of gates
- `gate_radius`: Gate radius in meters
- `turn_max_deg`: Max turn angle
- `episode_duration_s`: Episode timeout in seconds

### `evaluation` - Testing & Checkpoints
- `eval_every_steps`: Evaluation frequency
- `eval_episodes`: Episodes per evaluation
- `checkpoint_keep`: Keep last N checkpoints

### `curriculum` - Difficulty Progression
Define stages with step thresholds and environment overrides:
```yaml
curriculum:
  stages:
    - step: 0
      description: "Stage 1"
      overrides:
        gate_radius: 1.25
        n_gates: 3
    - step: 100_000
      description: "Stage 2"
      overrides:
        gate_radius: 1.0
        n_gates: 5
```

### `model` - Neural Network Architecture
- `cnn_channels`: Conv layer channel sizes
- `cnn_kernel`, `cnn_stride`: Conv layer parameters
- `imu_hidden`: IMU encoder hidden size
- `rnn_type`: "gru" or "lstm"
- `rnn_hidden`: RNN hidden size
- `actor_hidden`, `critic_hidden`: MLP head sizes

### `video` - Video Rendering
- `enabled`: Enable video recording
- `size`: Output resolution
- `fps`: Frame rate
- `overlay`: Debug text overlay
- `record_first_episode_only`: Only record first eval episode
- `quality`: "low", "medium", "high"
- `codec`: Video codec

### `logging` - Output & Diagnostics
- `tensorboard`: Write TensorBoard events
- `verbose`: Verbose console output
- `log_interval`: Log frequency

### `checkpointing` - Model Saving
- `save_interval`: Save frequency
- `save_on_eval`: Save after each evaluation

### `system` - Hardware & Performance
- `headless_gl`: "auto", "egl", "osmesa", or "off"
- `mujoco_threads`: Parallel threads per environment
- `pin_memory`: Pin PyTorch tensors to GPU

## Creating Custom Configs

1. Copy one of the existing configs as a template
2. Modify values as needed
3. Use it: `python3 scripts/train_recurrent_ppo.py --config config/my-config.yaml`

## Supported Formats

Configs can be in YAML or JSON format. Extensions determine the format:
- `.yaml` or `.yml` → YAML (requires PyYAML)
- `.json` → JSON (built-in)

To install YAML support:
```bash
pip install pyyaml
```
