# Tether Lift — Multi-Quadcopter Cooperative Payload Transport

A Drake-based simulation of **N quadcopters cooperatively lifting and transporting a payload** using flexible rope tethers. The system features cascaded PD control with **tension feedback for smooth payload pickup**, eliminating the jerky motion that occurs when ropes suddenly become taut.

![Simulation](https://img.shields.io/badge/Drake-C%2B%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project simulates a challenging robotics scenario: multiple quadcopters must coordinate to lift a heavy payload from the ground using flexible ropes. Key challenges addressed:

- **Slack-to-taut transition**: Ropes start slack; when they become taut, impulsive forces can destabilize the system
- **Load sharing**: Each quadcopter must carry its share of the payload weight
- **Formation flight**: Quadcopters maintain relative positions while following a trajectory
- **Distributed rope dynamics**: Bead-chain rope model captures swing and wave propagation

## Features

| Feature | Description |
|---------|-------------|
| **N Quadcopters** | Configurable number (default: 3) in circular formation |
| **Flexible Ropes** | Bead-chain model with 8 beads per rope (9 tension-only segments) |
| **Gaussian Rope Lengths** | Per-quadcopter mean and variance for uncertainty modeling |
| **Waypoint Trajectories** | Multi-phase: hover → ascend → translate → descend |
| **Tension Feedback** | Smooth pickup via ramped tension targets |
| **Real-time Visualization** | Meshcat 3D viewer with tension plots |

---

## Control Architecture

The controller implements a **cascaded PD structure** with tension feedback:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUADCOPTER CONTROLLER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Waypoint   │───►│    Position     │───►│      Attitude Controller    │ │
│  │  Trajectory │    │   Controller    │    │      (Inner Loop)           │ │
│  │  Generator  │    │  (Outer Loop)   │    │                             │ │
│  └─────────────┘    └────────┬────────┘    └──────────────┬──────────────┘ │
│                              │                            │                │
│                              │ thrust                     │ τx, τy, τz     │
│                              ▼                            ▼                │
│                     ┌────────────────────────────────────────────┐         │
│  Tension ──────────►│          Thrust Modification               │         │
│  Feedback           │  • Feedforward: thrust += T_measured       │         │
│  (via ZOH)          │  • Feedback: thrust += Kp*(T_target - T)   │         │
│                     │  • Altitude adjust: z_des += K_alt * e_T   │         │
│                     └────────────────────────────────────────────┘         │
│                                          │                                 │
│                                          ▼                                 │
│                              ┌────────────────────┐                        │
│                              │  SpatialForce      │                        │
│                              │  [thrust, τ]       │───► To Plant           │
│                              └────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Position Controller (Outer Loop)

**Purpose**: Track 3D waypoint trajectory with formation offset.

| Axis | Control Law | Gains | Output |
|------|-------------|-------|--------|
| **X** | `ax = Kp*(x_des - x) + Kd*(vx_des - vx)` | Kp=10, Kd=6 | Desired pitch angle |
| **Y** | `ay = Kp*(y_des - y) + Kd*(vy_des - vy)` | Kp=10, Kd=6 | Desired roll angle |
| **Z** | `az = Kp*(z_des - z) + Kd*(vz_des - vz)` | Kp=15, Kd=8 | Base thrust |

**X/Y to tilt conversion** (small angle approximation):
```cpp
pitch_des = clamp(ax / g, -max_tilt, max_tilt);  // +pitch → +x motion
roll_des  = clamp(-ay / g, -max_tilt, max_tilt); // +roll → -y motion
```

### 2. Attitude Controller (Inner Loop)

**Purpose**: Stabilize orientation and track desired roll/pitch from position controller.

| Axis | Control Law | Gains |
|------|-------------|-------|
| **Roll** | `τx = Kp*(roll_des - roll) - Kd*ωx` | Kp=8, Kd=1.5 |
| **Pitch** | `τy = Kp*(pitch_des - pitch) - Kd*ωy` | Kp=8, Kd=1.5 |
| **Yaw** | `τz = Kp*(0 - yaw) - Kd*ωz` | Kp=8, Kd=1.5 |

Angular velocities are computed in **body frame**: `ω_B = R^T * ω_W`

### 3. Tension Feedback Controller (Pickup Phase)

**Purpose**: Enable smooth payload pickup by gradually ramping tension instead of sudden load transfer.

#### Problem Without Tension Feedback
```
Time: ──────────────────►
       │
       │ Quad ascends
       │     ↓
       │ Rope becomes taut ─────► IMPULSE! Jerk, oscillation, potential instability
       │
```

#### Solution: 4-Stage Tension Control

**Stage 1: Pickup Detection**
```cpp
if (measured_tension >= 1.0 N) {
    pickup_start_time = current_time;  // Rope just became taut
}
```

**Stage 2: Tension Target Ramping**
```cpp
// Over 2 seconds, gradually increase target from 0 to full load
ramp_fraction = (t - pickup_start_time) / 2.0;  // 0 → 1
target_tension = ramp_fraction * (payload_weight / N_quads);
```

```
Target Tension
     ▲
     │            ╭─────────────── payload_weight/N
     │          ╱
     │        ╱
     │      ╱    (linear ramp over 2 seconds)
     │    ╱
     │  ╱
0 ───┼─╱─────────────────────────────────► Time
     │ t_taut     t_taut + 2s
```

**Stage 3: Thrust Feedforward**
```cpp
// Immediately compensate for rope tension pulling quad down
thrust += measured_tension;  // Feedforward
```

**Stage 4: Tension Feedback + Altitude Adjustment**
```cpp
tension_error = target_tension - measured_tension;

// Proportional feedback on thrust
thrust += 0.5 * tension_error;

// Altitude adjustment (if tension too low, move up to stretch rope more)
z_des += 0.003 * tension_error;  // Clamped to ±0.5m
```

#### Tension Signal Path

To avoid algebraic loops and simulate real sensor behavior, tension is passed through a **Zero-Order Hold**:

```
RopeForceSystem ──► ZeroOrderHold(dt) ──► Controller
                        │
                        └─ Holds tension value for one simulation timestep
                           Simulates sensor sampling delay
```

#### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pickup_detection_threshold` | 1.0 N | Tension level to trigger pickup mode |
| `pickup_ramp_duration` | 2.0 s | Time to ramp to full target tension |
| `pickup_target_tension` | `payload_weight / N` | Final tension target per rope |
| `tension_feedback_kp` | 0.5 | Proportional gain for tension error |
| `tension_altitude_gain` | 0.003 | Gain for altitude adjustment |
| `tension_altitude_max` | 0.5 m | Maximum altitude adjustment |

---

## Physical Model

### Quadcopters (6-DoF Free Rigid Bodies)

Each quadcopter is modeled as a **free-floating rigid body** with:
- Mass: 1.5 kg
- Dimensions: 0.30 × 0.30 × 0.10 m
- Control: Net thrust + net torque (abstracted rotor model)
- Visual: Drake quadrotor URDF mesh welded to rigid body

### Payload

- Mass: 3.0 kg
- Shape: Sphere (radius 0.15 m)
- Ground contact: Coulomb friction (μ_s=0.9, μ_d=0.7)
- Initial state: Resting on ground at origin

### Ropes (Bead-Chain Model)

Each rope consists of **8 beads** (small spheres) creating **9 segments**:

```
Quadcopter
    │
    ○ ─── segment 1 (tension-only spring-damper)
    │
    ○ ─── segment 2
    │
   ...
    │
    ○ ─── segment 8
    │
    ○ ─── segment 9
    │
 Payload
```

**Segment Physics** (tension-only spring-damper):
```cpp
stretch = distance - rest_length;
if (stretch <= 0) {
    force = 0;  // Slack: no compression forces
} else {
    tension = k * stretch;
    if (stretch_rate > 0) {
        tension += c * stretch_rate;  // Damping only when stretching
    }
    force = tension * unit_vector;
}
```

**Rope Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| Bead mass | 0.025 kg | Per bead |
| Bead radius | 0.02 m | For collision |
| Segment stiffness | ~200 N/m | Computed from desired stretch |
| Segment damping | ~15 N·s/m | Scaled with √stiffness |
| Max stretch | 15% | Design target |

### Gaussian Rope Length Distribution

Rope lengths are sampled from **per-quadcopter Gaussian distributions** to model measurement uncertainty:

```cpp
rope_length_means   = {1.0, 1.1, 0.95};  // Expected lengths [m]
rope_length_stddevs = {0.05, 0.08, 0.06}; // Uncertainties [m]

// Sample: rope_length[i] ~ N(mean[i], stddev[i]²)
```

Example output:
```
Quad 0: mean=1.0m, stddev=0.05m -> sampled=0.914m
Quad 1: mean=1.1m, stddev=0.08m -> sampled=1.105m
Quad 2: mean=0.95m, stddev=0.06m -> sampled=0.995m
```

---

## Trajectory System

### Waypoint-Based Trajectories

The shared trajectory is defined as a sequence of waypoints:

```cpp
struct TrajectoryWaypoint {
    Eigen::Vector3d position;  // Target [x, y, z] in world frame
    double arrival_time;       // When to reach this waypoint
    double hold_time;          // How long to hover before next
};
```

**Default Trajectory**:
| Phase | Position | Arrival | Hold | Description |
|-------|----------|---------|------|-------------|
| 1 | (0, 0, 1.2) | 0s | 1s | Initial hover |
| 2 | (0, 0, 3.0) | 4s | 2s | Ascend (lift payload) |
| 3 | (2, 1, 3.0) | 8s | 2s | Translate horizontally |
| 4 | (2, 1, 2.0) | 12s | 3s | Descend to final position |

### Formation Offsets

Each quadcopter adds its **formation offset** to the shared trajectory:

```cpp
// Circular formation around trajectory center
angle = 2π * i / N_quads;
formation_offset[i] = (radius * cos(angle), radius * sin(angle), 0);

// Actual position = trajectory_position + formation_offset
```

---

## Visualization

### Meshcat 3D Viewer

Open http://localhost:7000 to see:

- **Quadcopters**: URDF models with rotors
- **Payload**: Red sphere
- **Ropes**: Colored polylines (blue, green, yellow for quads 0, 1, 2)
- **Ground**: Gray plane
- **Reference Trajectory**: Green line showing expected load path
- **Actual Trails**: Orange (load) and colored (drones) position history

### Real-Time Tension Plot

A 3D line graph in the corner shows tension history:
- One trace per rope (color-coded)
- Vertical bars showing current tension magnitude
- Auto-scaling based on maximum observed tension

---

## Data Logging

All simulation data is automatically logged to timestamped CSV files:

```
outputs/logs/YYYYMMDD_HHMMSS/
├── config.txt           # Simulation parameters
├── trajectories.csv     # Ground truth positions/velocities
├── tensions.csv         # Rope tensions
├── control_efforts.csv  # Controller torques/forces
├── gps_measurements.csv # Noisy GPS readings
├── estimator_outputs.csv # State estimates
└── reference_trajectory.csv # Desired trajectory
```

Each run creates a new folder, enabling easy comparison between experiments.

See [cpp/README.md](cpp/README.md) for detailed file format documentation.

---

## Project Structure

```
Tether_Lift/
├── README.md                   # This file
├── outputs/
│   └── logs/                   # Timestamped simulation data
├── cpp/                        # C++ implementation
│   ├── CMakeLists.txt
│   ├── README.md               # Build instructions
│   ├── include/
│   │   ├── quadcopter_controller.h
│   │   ├── rope_force_system.h
│   │   ├── rope_utils.h
│   │   ├── rope_visualizer.h
│   │   ├── tension_plotter.h
│   │   ├── trajectory_visualizer.h
│   │   └── simulation_data_logger.h
│   └── src/
│       ├── main.cc
│       ├── quadcopter_controller.cc
│       ├── rope_force_system.cc
│       ├── rope_utils.cc
│       ├── rope_visualizer.cc
│       ├── tension_plotter.cc
│       ├── trajectory_visualizer.cc
│       └── simulation_data_logger.cc
├── scripts/                    # Python prototype
│   └── quad_rope_lift.py
├── drake/                      # Drake source (submodule)
└── References/                 # Design documents
```

---

## Quick Start

```bash
# Build
cd cpp
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_PREFIX_PATH=/opt/drake
ninja

# Run
./quad_rope_lift

# Open visualization
# Navigate to http://localhost:7000 in your browser
```

---

## Configuration

Key parameters in `cpp/src/main.cc`:

```cpp
// Simulation
const double simulation_time_step = 2e-4;  // [s]
const double simulation_duration = 15.0;   // [s]

// Multi-quadcopter
const int num_quadcopters = 3;
const double formation_radius = 0.5;  // [m]

// Physical properties
const double quadcopter_mass = 1.5;   // [kg]
const double payload_mass = 3.0;      // [kg]
const int num_rope_beads = 8;

// Rope length distributions (per quadcopter)
std::vector<double> rope_length_means = {1.0, 1.1, 0.95};
std::vector<double> rope_length_stddevs = {0.05, 0.08, 0.06};
const unsigned int random_seed = 42;  // For reproducibility
```

---

## Dependencies

- **Drake** (v1.25.0+): Robotics simulation library
- **Eigen3**: Linear algebra
- **CMake** (3.16+): Build system
- **C++20**: Language standard

The included DevContainer provides all dependencies pre-installed.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Drake](https://drake.mit.edu/) - Robotics simulation toolkit
- Design approach inspired by tension-aware control literature for cable-suspended robots
