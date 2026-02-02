# Multi-Quadcopter Rope Lift - C++ Implementation

Drake simulation of multiple quadcopters cooperatively lifting a payload using flexible ropes/tethers.

## Features

- **N quadcopters** in configurable formation (default: 3 in triangular arrangement)
- **Independent ropes** connecting each quadcopter to the payload
- **Waypoint-based trajectory** with hover → ascend → translate phases
- **Tension-aware pickup** with load sharing across all quadcopters
- **Real-time visualization** via Meshcat

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
├── CMakeLists.txt              # Build configuration (C++20, Drake, Eigen3)
├── README.md                   # This file
├── include/
│   ├── quadcopter_controller.h # Position/attitude controller with waypoint trajectory
│   ├── rope_force_system.h     # Tension-only spring-damper rope physics
│   ├── rope_utils.h            # Rope parameter calculation & slack initialization
│   └── rope_visualizer.h       # Meshcat polyline visualization
└── src/
    ├── main.cc                 # Multi-quad simulation setup and main loop
    ├── quadcopter_controller.cc
    ├── rope_force_system.cc
    ├── rope_utils.cc
    └── rope_visualizer.cc
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
quad_configs[i].rope_length = 1.0;    // [m] - can vary per quad
```

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

## How It Works

1. **Initialization**: N quadcopters arranged in circle around payload, each with slack rope
2. **Hover**: All quadcopters stabilize at initial altitude
3. **Ascent**: Formation rises, ropes become taut, tension detected
4. **Pickup**: Controllers ramp up thrust to smoothly lift payload
5. **Translation**: Formation moves horizontally while maintaining payload
6. **Final Position**: System stabilizes at destination

## Output

```
Starting multi-quadcopter simulation...
  Number of quadcopters: 3
  Payload mass: 3 kg
  Load per quadcopter: 1 kg
  Duration: 15 s
Open Meshcat at: http://localhost:7000

  Simulated 1s / 15s
  ...

Simulation complete!
  Max rope tension (rope 0): 12.5 N
  Expected tension per rope: 9.81 N
  Total payload weight: 29.43 N
```
