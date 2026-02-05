# Multi-Quadcopter Rope Lift - C++ Implementation

Drake simulation of multiple quadcopters cooperatively lifting a payload using flexible ropes/tethers.

## Features

### Core Simulation
- **N quadcopters** in configurable formation (default: 3 in triangular arrangement)
- **Independent ropes** connecting each quadcopter to the payload
- **Gaussian-distributed rope lengths** with per-quadcopter mean and variance for uncertainty modeling
- **Waypoint-based trajectory** with hover → ascend → translate phases
- **Tension-aware pickup** with load sharing across all quadcopters
- **Real-time tension plots** in Meshcat with 3D line graphs and bar indicators
- **Real-time visualization** via Meshcat with colored ropes per quadcopter

### GPAC: Geometric Position and Attitude Control (NEW)
- **Layer 1 - Position + Anti-swing Control** (50 Hz)
  - PID position tracking with integral anti-windup
  - S² cable direction control [Eq. 8-9] for anti-swing
  - ESO disturbance feedforward compensation
- **Layer 2 - Geometric Attitude Control** (200 Hz)
  - SO(3) attitude error using geometric formulation [Eq. 12]
  - Desired rotation from thrust direction [Eq. 19-20]
  - Proper angular velocity error tracking
- **Layer 3 - Concurrent Learning Estimator** (200 Hz)
  - Adaptive load mass estimation
  - History stack with rank-maximizing data selection
  - Guaranteed convergence without persistent excitation
- **Layer 4 - Extended State Observer** (500 Hz)
  - Third-order ESO for disturbance estimation
  - Per-axis observers with configurable bandwidth
  - Lumped uncertainty compensation
- **CBF Safety Filter** - Tautness constraints for cable-suspended loads
  - Tension bounds (min/max)
  - Cable angle limits
  - Swing rate constraints
  - Quadcopter tilt limits

### Sensor Suite (Phase 1-2)
- **IMU Sensor** (200 Hz)
  - 6-DOF: 3-axis gyroscope + 3-axis accelerometer
  - Gauss-Markov bias dynamics with configurable time constants
  - Consumer-grade noise: gyro 0.0005 rad/s/√Hz, accel 0.004 m/s²/√Hz
  - Numerical acceleration via velocity differentiation
- **Barometer Sensor** (25 Hz)
  - Three-component noise: white (0.3m) + correlated (0.2m, τ=5s) + drift
  - Altitude quantization (0.1m resolution)
- **GPS Sensor** (10 Hz)
  - Position noise: 0.02m (x,y), 0.05m (z)
  - Validity flag for dropout modeling
- **Wind Disturbance System** (100 Hz)
  - Dryden turbulence model with spatial correlation
  - Mean wind: [1.0, 0.5, 0.0] m/s (configurable)
  - Turbulence intensities: σu=0.5, σv=0.5, σw=0.25 m/s
  - Altitude-dependent scaling

### State Estimation
- **15-State ESKF** - Error-State Kalman Filter fusing IMU/GPS/Barometer
  - State: [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]
  - Multiplicative quaternion error representation (MEKF)
- **Position/Velocity Estimator** - Legacy 6-state EKF
- **Load Estimator** - EKF with taut-gated cable constraints

### Decentralized Adaptive Control (Phase 3-4)
- **Adaptive Load Estimator** - Concurrent learning for θ̂ = m_L/N estimation
  - No knowledge of N (number of quads) or m_L (load mass) required
  - Converges without persistent excitation (finite excitation sufficient)
- **Decentralized Load State Estimator** - Each quad estimates load position locally
  - Uses cable geometry: p_L = p_Q - L * n
  - Tension-based confidence weighting
- **N-Independent Controller** - Uses adaptive θ̂ for feedforward instead of m_L/N

### Trajectory and Safety (Phase 5-7)
- **Load-Centric Trajectory Generator** - Minimum-jerk smooth trajectories
  - 12D output: [position, velocity, acceleration, jerk]
  - Feasibility checking for horizontal acceleration limits
  - Waypoint interpolation with hold times
- **Required Force Computer** - Computes F_req from load tracking error
- **Force Allocation System** - Equal-share allocation to drones
- **Drone Trajectory Mapper** - Maps load trajectory to per-drone via cable geometry
- **CBF Safety Filter** - Control Barrier Functions for constraint enforcement
  - Cable tension bounds (min/max)
  - Cable angle limits
  - Butterworth-filtered tension rate estimation
- **Wind Disturbance Model** - Dryden turbulence with spatial correlation

### Visualization & Data Logging
- **Trajectory Visualizer** - Real-time Meshcat visualization
  - Green reference trajectory line showing expected load path
  - Orange trail showing actual load position history
  - Colored trails for each drone's path
- **Comprehensive Data Logger** - Timestamped CSV output (12 files)
  - Output: `/workspaces/Tether_Lift/outputs/logs/YYYYMMDD_HHMMSS/`
  - 100 Hz logging with automatic buffer flushing
  - Files: trajectories, imu, barometer, gps, estimator, tensions, control, attitude, wind, config

## Building

### Prerequisites

The devcontainer already includes:
- CMake 3.16+
- C++20 compiler (GCC or Clang)
- Drake C++ library (installed at `/opt/drake`)
- Eigen3

### Build Instructions

```bash
# From the cpp directory
cd cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake (using Ninja for faster builds)
cmake .. -G Ninja -DCMAKE_PREFIX_PATH=/opt/drake

# Build
ninja

# Or with Make:
cmake .. -DCMAKE_PREFIX_PATH=/opt/drake
make -j$(nproc)
```

### Running

```bash
./quad_rope_lift
```

The simulation opens a Meshcat visualizer at http://localhost:7000

## Project Structure

```
cpp/
├── CMakeLists.txt                  # Build configuration (C++20, Drake, Eigen3)
├── README.md                       # This file
├── include/
│   │
│   │ # Core Simulation
│   ├── quadcopter_controller.h     # Position/attitude controller with waypoint trajectory
│   ├── rope_force_system.h         # Tension-only spring-damper rope physics
│   ├── rope_utils.h                # Rope parameter calculation & slack initialization
│   ├── rope_visualizer.h           # Meshcat polyline visualization
│   ├── tension_plotter.h           # Real-time tension graph visualization
│   │
│   │ # GPAC: Geometric Position and Attitude Control (Layers 1-4)
│   ├── gpac_math.h                 # SO(3)/S² operations: Hat, Vee, rotation errors
│   ├── gpac_load_tracking_controller.h # Layer 1: Position + S² anti-swing (50 Hz)
│   ├── gpac_quadcopter_controller.h    # Layer 2: Geometric SO(3) attitude (200 Hz)
│   ├── concurrent_learning_estimator.h # Layer 3: Adaptive mass estimation (200 Hz)
│   ├── extended_state_observer.h       # Layer 4: Disturbance estimation (500 Hz)
│   ├── gpac_cbf_safety_filter.h        # CBF tautness constraints
│   │
│   │ # Sensors (Phase 1)
│   ├── imu_sensor.h                # 6-DOF IMU with Gauss-Markov bias (200 Hz)
│   ├── barometer_sensor.h          # Altitude with 3-component noise (25 Hz)
│   ├── gps_sensor.h                # GPS position with noise (10 Hz)
│   │
│   │ # State Estimation (Phase 2)
│   ├── eskf_estimator.h            # 15-state Error-State Kalman Filter
│   ├── position_velocity_estimator.h  # Legacy 6-state EKF for quadcopters
│   ├── load_estimator.h            # Legacy EKF with taut-gated constraints
│   ├── estimation_utils.h          # Helper systems (position extractors, etc.)
│   ├── estimation_error_computer.h # Ground truth comparison
│   │
│   │ # Decentralized Adaptive Control (Phase 3-4)
│   ├── adaptive_load_estimator.h   # Concurrent learning θ̂ = m_L/N estimation
│   ├── decentralized_load_estimator.h # Local load state per quad
│   ├── adaptive_lift_controller.h  # N-independent controller using θ̂
│   │
│   │ # Trajectory and Safety (Phase 5-7)
│   ├── load_trajectory_generator.h # Min-jerk load-centric trajectories
│   ├── cbf_safety_filter.h         # Control Barrier Function safety
│   ├── wind_disturbance.h          # Dryden turbulence wind model
│   │
│   │ # Visualization (No .h in include - defined in src/)
│   └── (visualization headers in src/)
│
└── src/
    ├── main.cc                     # Multi-quad simulation setup and main loop
    │
    │ # Core
    ├── quadcopter_controller.cc
    ├── rope_force_system.cc
    ├── rope_utils.cc
    ├── rope_visualizer.cc
    ├── tension_plotter.cc
    │
    │ # GPAC Components
    ├── gpac_quadcopter_controller.cc   # Layer 2: Geometric attitude control
    ├── gpac_load_tracking_controller.cc # Layer 1: Position + anti-swing
    ├── concurrent_learning_estimator.cc # Layer 3: Adaptive estimation
    ├── extended_state_observer.cc      # Layer 4: ESO implementation
    ├── gpac_cbf_safety_filter.cc       # CBF tautness filter
    │
    │ # Sensors
    ├── imu_sensor.cc
    ├── barometer_sensor.cc
    ├── gps_sensor.cc
    │
    │ # State Estimation
    ├── eskf_estimator.cc
    ├── position_velocity_estimator.cc
    ├── load_estimator.cc
    ├── estimation_utils.cc
    ├── estimation_error_computer.cc
    │
    │ # Adaptive Control
    ├── adaptive_load_estimator.cc
    ├── decentralized_load_estimator.cc
    ├── adaptive_lift_controller.cc
    │
    │ # Trajectory & Safety
    ├── load_trajectory_generator.cc
    ├── cbf_safety_filter.cc
    ├── wind_disturbance.cc
    │
    │ # Load Trajectory Following Pipeline
    ├── required_force_computer.h/cc    # F_req from load tracking error
    ├── force_allocation_system.h/cc    # Equal-share allocation
    ├── drone_trajectory_mapper.h/cc    # Load → drone trajectory
    ├── extended_load_trajectory_generator.h/cc # 12D trajectory [p,v,a,j]
    ├── load_tracking_controller.h/cc   # Full load tracking control
    ├── decentralized_drone_controller.h/cc # N-independent controller
    │
    │ # Visualization & Logging
    ├── trajectory_visualizer.h/cc      # Reference + actual trajectory viz
    └── simulation_data_logger.h/cc     # Comprehensive CSV logging (12 files)
```

## Physical Model

- **Quadcopters**: N free-floating 6-DoF rigid bodies with thrust and torque control
- **Payload**: Spherical rigid body resting on ground with Coulomb friction
- **Ropes**: N independent bead-chain models, each with tension-only spring-damper segments
- **Ground**: Infinite half-space with friction contact

## Control Strategy

1. **Position Controller**: PD control tracking 3D waypoint trajectory with formation offset
2. **Attitude Controller**: PD control tracking desired roll/pitch (from x/y position error) and zero yaw
3. **Tension-Aware Pickup**: Detects rope becoming taut and ramps up tension feedforward
4. **Load Sharing**: Each quadcopter targets `payload_weight / N` tension

## Configuration

Key parameters in `src/main.cc`:

```cpp
// Simulation
const double simulation_time_step = 2e-4;  // [s]
const double simulation_duration = 15.0;   // [s]

// Number of quadcopters
const int num_quadcopters = 3;

// Formation geometry
const double formation_radius = 0.5;  // [m] - horizontal distance from payload

// Quadcopter (same for all)
const double quadcopter_mass = 1.5;  // [kg]

// Payload
const double payload_mass = 3.0;    // [kg]
const double payload_radius = 0.15; // [m]

// Rope (per quadcopter)
const double rope_total_mass = 0.2;   // [kg]
const int num_rope_beads = 8;
```

### Gaussian Rope Length Distribution

Rope lengths are sampled from Gaussian distributions to model measurement uncertainty:

```cpp
// Expected rope lengths for each quadcopter [m]
std::vector<double> rope_length_means = {1.0, 1.1, 0.95};

// Standard deviations representing measurement uncertainty [m]
std::vector<double> rope_length_stddevs = {0.05, 0.08, 0.06};

// Random seed for reproducibility (change for different realizations)
const unsigned int random_seed = 42;
```

Each `quad_configs[i]` stores:
- `rope_length_mean` - expected rope length
- `rope_length_stddev` - measurement uncertainty (standard deviation)
- `rope_length` - actual sampled value from N(mean, stddev²)

### Trajectory Waypoints

The shared trajectory is defined as a sequence of waypoints:

```cpp
std::vector<TrajectoryWaypoint> waypoints;

// Phase 1: Hover at initial position
waypoints.push_back({Eigen::Vector3d(0, 0, 1.2), 0.0, 1.0});

// Phase 2: Ascend to lift payload
waypoints.push_back({Eigen::Vector3d(0, 0, 3.0), 4.0, 2.0});

// Phase 3: Translate horizontally
waypoints.push_back({Eigen::Vector3d(2.0, 1.0, 3.0), 8.0, 2.0});

// Phase 4: Descend to final position
waypoints.push_back({Eigen::Vector3d(2.0, 1.0, 2.0), 12.0, 3.0});
```

Each waypoint has:
- `position`: Target (x, y, z) in world frame
- `arrival_time`: When to reach this waypoint
- `hold_time`: How long to hover before moving to next

Each quadcopter adds its `formation_offset` to maintain formation.

## Decentralized Control Architecture

The decentralized control architecture enables each quadcopter to operate independently without knowledge of:
- **N** - Total number of quadcopters
- **m_L** - Load mass

### Key Insight

Instead of using `pickup_target_tension = m_L * g / N` (which requires centralized knowledge), each quadcopter estimates θ̂ᵢ ≈ m_L/N through local observations.

### Adaptive Load Estimator

From load dynamics: `m_L * (a_L + g) = Σᵢ Tᵢ * nᵢ`

Each drone observes:
- Cable tension Tᵢ (from sensor or model)
- Cable direction nᵢ (from geometry)
- Load acceleration estimate (from velocity differentiation)

**Concurrent Learning Adaptive Law:**
```
θ̇̂ᵢ = -Γ Yᵢᵀ(t) s(t) - Γ Σⱼ Yᵢᵀ(tⱼ) εⱼ(t)
```

Where:
- Yᵢ = ||a_L + g|| is the regressor
- s is sliding variable (tracking error + λ × position error)
- εⱼ are prediction errors on stored historical data

**Convergence:** θ̂ᵢ → m_L/N without requiring persistent excitation.

### N-Independent Controller

The adaptive controller uses θ̂ for feedforward:

```cpp
// OLD (centralized):
target_tension = payload_weight / N;

// NEW (decentralized):
target_tension = theta_hat * gravity;  // θ̂ ≈ m_L/N
```

**Control Law:**
```
F_quad = m_Q * (g + a_position) + θ̂ * g * n + F_tension_regulation
```

### Safety Guarantees (CBF)

Control Barrier Functions ensure:
1. **Tension bounds:** T_min ≤ T ≤ T_max
2. **Cable angles:** θ ≤ θ_max from vertical
3. **Collision avoidance:** ||rᵢ - rⱼ|| ≥ d_min

## How It Works

1. **Initialization**: N quadcopters arranged in circle around payload, each with slack rope
2. **Hover**: All quadcopters stabilize at initial altitude
3. **Ascent**: Formation rises, ropes become taut, tension detected
4. **Pickup**: Controllers ramp up thrust to smoothly lift payload
5. **Translation**: Formation moves horizontally while maintaining payload
6. **Final Position**: System stabilizes at destination

## Output

```
========================================
Rope Length Sampling (Gaussian)
========================================
Random seed: 42
Quad 0: mean=1m, stddev=0.05m -> sampled=0.914294m
Quad 1: mean=1.1m, stddev=0.08m -> sampled=1.10457m
Quad 2: mean=0.95m, stddev=0.06m -> sampled=0.995377m
========================================

Average sampled rope length: 1.00475m


========================================
State Estimation Enabled
========================================
GPS sample rate: 10 Hz
GPS noise (x,y,z): 0.02 0.02 0.05 m
Estimator rate: 100 Hz
Use estimated in controller: NO
========================================


========================================
IMU Sensors Enabled
========================================
IMU sample rate: 200 Hz
Gyro noise density: 0.0005 0.0005 0.0005 rad/s/sqrt(Hz)
Accel noise density: 0.004 0.004 0.004 m/s^2/sqrt(Hz)
========================================


========================================
Barometer Sensors Enabled
========================================
Barometer sample rate: 25 Hz
White noise stddev: 0.3 m
Correlated noise stddev: 0.2 m
========================================


========================================
Wind Disturbance Enabled
========================================
Mean wind:   1 0.5   0 m/s
Turbulence (u,v,w): 0.5, 0.5, 0.25 m/s
Gusts enabled: NO
========================================

[Meshcat listening for connections at http://localhost:7000]
Data logger initialized. Output directory: /workspaces/Tether_Lift/outputs/logs/20260204_185348

Configuration saved to: /workspaces/Tether_Lift/outputs/logs/20260204_185348/config.txt

  Simulated 1s / 15s
  ...

Simulation complete!
  Max rope tension (rope 0): 12.5 N
  Expected tension per rope: 9.81 N
  Total payload weight: 29.43 N
```

## Visualization

The Meshcat visualizer (http://localhost:7000) displays:

- **Quadcopters**: URDF models with rotors
- **Payload**: Red sphere
- **Ropes**: Colored polylines (blue, green, yellow for quads 0, 1, 2)
- **Tension Plot**: 3D line graph in corner showing real-time tension history
  - One trace per rope (color-coded)
  - Vertical bars showing current tension magnitude
  - Auto-scaling based on maximum tension
- **Reference Trajectory**: Green line showing expected load path
- **Actual Trails**: Orange (load) and colored (drones) trails showing traversed paths

## State Estimation System

The simulation includes an optional GPS-based state estimation system for position and velocity:

### Components

1. **GPS Sensor** (`gps_sensor.h/.cc`)
   - Simulates noisy GPS position measurements
   - Configurable sample rate (default: 10 Hz)
   - Gaussian noise model with per-axis standard deviations
   - Optional dropout probability for realistic dropouts

2. **Position/Velocity Estimator** (`position_velocity_estimator.h/.cc`)
   - Extended Kalman Filter (EKF) for quadcopter state estimation
   - 6-state model: [position_x/y/z, velocity_x/y/z]
   - Constant velocity process model with configurable noise
   - GPS position measurement updates when valid

3. **Load State Estimator** (`load_estimator.h/.cc`)
   - EKF for payload state estimation
   - GPS + cable constraint pseudo-measurements
   - Taut-gated constraints: only applies cable length constraints when tension > threshold
   - Uses known cable lengths and quadcopter attachment positions

4. **Helper Systems** (`estimation_utils.h/.cc`)
   - `AttachmentPositionExtractor`: Extracts quad attachment points from plant state
   - `CableLengthSource`: Provides known cable lengths to load estimator
   - `TensionAggregator`: Aggregates tension from all rope systems

5. **Estimation Error Computer** (`estimation_error_computer.h/.cc`)
   - Computes error between estimated and ground truth states
   - Outputs: position error (x/y/z), velocity error (x/y/z), norm errors

### Configuration

```cpp
// Enable GPS-based state estimation
const bool enable_estimation = true;

// Use estimated state in controller (requires IMU for stability)
const bool use_estimated_in_controller = false;

// GPS sensor parameters
const double gps_sample_period = 0.1;  // [s] (10 Hz)
const Eigen::Vector3d gps_position_noise(0.02, 0.02, 0.05);  // [m] stddev

// Estimator update rate
const double estimator_dt = 0.01;  // [s] (100 Hz)
```

### Performance

With default settings (GPS noise: 2cm x/y, 5cm z):
- **Quadcopter position estimation**: ~7 cm RMS, ~20 cm max error
- **Load position estimation**: ~37 cm RMS, ~45 cm max error
  - Higher error because cable constraints only active when taut

### Architecture Notes

- The estimator runs in "open loop" mode by default (`use_estimated_in_controller = false`)
- Using estimated state in the controller without IMU causes instability due to low GPS rate
- Future work: Add IMU sensors for high-rate propagation to enable closed-loop control

### Output

```
========================================
State Estimation Enabled
========================================
GPS sample rate: 10 Hz
GPS noise (x,y,z): 0.02 0.02 0.05 m
Estimator rate: 100 Hz
Use estimated in controller: NO
========================================

...

========================================
Estimation Error Statistics
========================================
Quad 0: RMS pos error = 7.04 cm, Max = 18.33 cm
Quad 1: RMS pos error = 7.24 cm, Max = 23.76 cm
Quad 2: RMS pos error = 6.54 cm, Max = 21.47 cm
Load:   RMS pos error = 37.47 cm, Max = 44.73 cm
========================================
```

## Data Logging

The simulation automatically logs all important signals to timestamped CSV files for post-processing and analysis.

### Output Location

Data is saved to `/workspaces/Tether_Lift/outputs/logs/<YYYYMMDD_HHMMSS>/`

Each run creates a new folder with the timestamp, allowing easy comparison between runs.

### Logged Files

| File | Contents | Columns |
|------|----------|---------|
| `config.txt` | Simulation parameters | Key-value pairs |
| `trajectories.csv` | Ground truth positions/velocities | time, load_xyz, load_vxyz, droneN_xyz, droneN_vxyz |
| `tensions.csv` | Rope tensions | time, ropeN_mag, ropeN_fx/fy/fz |
| `control_efforts.csv` | Controller outputs | time, droneN_tau_xyz, droneN_f_xyz |
| `gps_measurements.csv` | Noisy GPS readings | time, load_gps_xyz, droneN_gps_xyz |
| `estimator_outputs.csv` | State estimates | time, load_est_xyz_vxyz, droneN_est_xyz_vxyz |
| `reference_trajectory.csv` | Desired trajectory | time, ref_xyz, ref_vxyz, ref_axyz |

### Configuration File Example

```
# Simulation Configuration
# Generated at: 20260203_070642

simulation_time_step = 0.000200
simulation_duration = 15.000000
num_quadcopters = 3
quadcopter_mass = 1.500000
payload_mass = 3.000000
avg_rope_length = 1.004749
segment_stiffness = 585.818201
enable_estimation = true

quad_0_rope_length_mean = 1.000000
quad_0_rope_length_sampled = 0.914294
...

waypoint_0_position = (0.000000, 0.000000, 1.200000)
waypoint_0_arrival_time = 0.000000
waypoint_0_hold_time = 1.000000
...
```

### Logging Rate

Data is logged at 100 Hz (configurable via `log_period` parameter). Automatic buffer flushing every 10 samples ensures data integrity even if simulation is interrupted.

### Log Files Generated

| File | Content | Columns |
|------|---------|---------|
| `trajectories.csv` | Ground truth state | time, load_[x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz], drone{N}_[...] |
| `imu_measurements.csv` | 6-DOF IMU data | time, drone{N}_[ax,ay,az,wx,wy,wz] |
| `barometer_measurements.csv` | Altitude readings | time, drone{N}_altitude |
| `gps_measurements.csv` | GPS positions | time, load_gps_[x,y,z], drone{N}_gps_[x,y,z] |
| `estimator_outputs.csv` | EKF/ESKF estimates | time, load_est_[x,y,z,vx,vy,vz], drone{N}_est_[...] |
| `tensions.csv` | Rope forces | time, rope{N}_[mag,fx,fy,fz] |
| `control_efforts.csv` | Control outputs | time, drone{N}_[tau_x,tau_y,tau_z,f_x,f_y,f_z] |
| `attitude_data.csv` | Orientation data | time, drone{N}_[roll,pitch,yaw,des_qw,...,err_x,err_y,err_z] |
| `wind_disturbance.csv` | Wind velocities | time, wind_[vx,vy,vz] |
| `config.txt` | Parameters | Key-value pairs |

### Post-Processing

Example Python analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
traj = pd.read_csv('outputs/logs/20260203_070642/trajectories.csv')
tensions = pd.read_csv('outputs/logs/20260203_070642/tensions.csv')

# Plot load trajectory
fig, ax = plt.subplots()
ax.plot(traj['time'], traj['load_z'], label='Actual')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Load Z [m]')
ax.legend()
plt.show()

# Plot tensions
fig, ax = plt.subplots()
for i in range(3):
    ax.plot(tensions['time'], tensions[f'rope{i}_mag'], label=f'Rope {i}')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Tension [N]')
ax.legend()
plt.show()
```

## Trajectory Visualization

The Meshcat visualizer shows both reference and actual trajectories:

### Reference Trajectory (Green Line)
- Shows expected load path based on drone waypoints
- Accounts for:
  - Rope attachment offsets (drone and payload)
  - Rope stretch under load (~15%)
  - Average rope length across all drones

### Actual Trails
- **Orange**: Load center position history
- **Colored**: Drone position histories (blue, magenta, purple, cyan, yellow, red cycle)
- Trails update at 30 Hz with configurable max points

### Reference Calculation

The load reference position is computed as:
```
load_z = drone_z - total_vertical_offset

where:
  total_vertical_offset = |quad_attachment_offset.z|   # ~0.10m
                        + rope_length × stretch_factor  # ~1.15m
                        + payload_radius                # ~0.15m
```

This ensures the green reference line closely matches where the load actually travels.
