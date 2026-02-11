"""Global constants for RL drone hoops project."""

# Environment constants
DEFAULT_IMAGE_SIZE = 96
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_PHYSICS_HZ = 1000.0
DEFAULT_CAMERA_FPS = 60.0
DEFAULT_IMU_HZ = 400.0

# Drone dynamics
MAX_TILT_DEG = 75.0  # Maximum allowed roll/pitch before termination (degrees)

# Action space
ACTION_DIM = 4  # Dimension of action space: [roll_rate, pitch_rate, yaw_rate, thrust]

# Reward shaping limits
MIN_REWARD_WEIGHT = 0.0
MAX_REWARD_WEIGHT = 1000.0  # Sanity bounds for reward weights

# Centering penalty falloff distance (meters). The centering penalty is
# attenuated linearly so it only applies when the drone is within this
# distance of the gate, avoiding large penalties when far away.
CENTER_FALLOFF_DISTANCE = 5.0

# Numerical stability
EPSILON = 1e-6  # Small epsilon for numerical stability

EPS_GRAD_UNDERFLOW = 20.0  # Clamp log-ratios to [-EPS_GRAD_UNDERFLOW, +EPS_GRAD_UNDERFLOW] for gradient stability

# Renderer cache
MAX_CACHED_RENDERERS = 5  # Maximum number of renderers to cache at different resolutions

# XML element names
XML_CAMERA_NAME = "fpv"
XML_DRONE_BODY_NAME = "drone"
XML_GATE_BODY_PREFIX = "gate"
XML_GATE_SEG_PREFIX = "seg"
