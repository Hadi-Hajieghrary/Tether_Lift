# Quadcopter Rope Lift - C++ Implementation

Drake simulation of a quadcopter lifting a payload using a flexible rope/tether.

## Building

### Prerequisites

The devcontainer already includes:
- CMake 3.16+
- C++20 compiler (GCC or Clang)
- Drake C++ library (installed at `/opt/drake`)

### Build Instructions

```bash
# From the workspace root
cd cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DCMAKE_PREFIX_PATH=/opt/drake

# Build
cmake --build . -j$(nproc)

# Or use Ninja for faster builds:
cmake .. -G Ninja -DCMAKE_PREFIX_PATH=/opt/drake
ninja
```

### Running

```bash
./build/quad_rope_lift
```

The simulation opens a Meshcat visualizer at http://localhost:7000

## Project Structure

```
cpp/
├── CMakeLists.txt           # Build configuration
├── include/
│   ├── quadcopter_controller.h  # PD altitude/attitude controller
│   ├── rope_force_system.h      # Spring-damper rope physics
│   ├── rope_utils.h             # Rope parameter calculation
│   └── rope_visualizer.h        # Meshcat polyline visualization
└── src/
    ├── main.cc                  # Simulation setup and main loop
    ├── quadcopter_controller.cc
    ├── rope_force_system.cc
    ├── rope_utils.cc
    └── rope_visualizer.cc
```

## Physical Model

- **Quadcopter**: Free-floating 6-DoF rigid body with thrust and torque control
- **Payload**: Spherical rigid body resting on ground with friction
- **Rope**: Bead-chain model with tension-only spring-damper segments

## Control Strategy

1. **Altitude Controller**: PD control following height trajectory
2. **Attitude Controller**: PD control maintaining upright orientation
3. **Tension-Aware Pickup**: Feedforward tension compensation for smooth load transfer

## Configuration

Key parameters can be modified in `main.cc`:

```cpp
// Physics
const double simulation_time_step = 2e-4;  // [s]
const double simulation_duration = 8.0;     // [s]

// Quadcopter
const double quadcopter_mass = 1.5;  // [kg]

// Payload
const double payload_mass = 2.0;    // [kg]
const double payload_radius = 0.12; // [m]

// Rope
const double rope_rest_length = 1.0;  // [m]
const int num_rope_beads = 10;
const double max_stretch_percentage = 0.15;  // 15%
```
