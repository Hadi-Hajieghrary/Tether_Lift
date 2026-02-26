# Research — Simulation Code & Scripts

This directory contains the core simulation code for the **Multi-Quadcopter Cooperative Payload Transport** project.

See the [main project README](../README.md) for a full overview of the GPAC architecture, physical model, and control design.

## Directory Structure

```
Research/
├── cpp/                        # C++ Drake simulation
│   ├── CMakeLists.txt
│   ├── README.md               # Build instructions & detailed docs
│   ├── include/                # Public headers (25 files)
│   ├── src/                    # Source files + private headers
│   │   ├── main.cc             # Simulation entry point
│   │   └── ... (33 .cc files, 9 private .h files)
│   └── gpac/                   # GPAC test harness
│       ├── src/main_gpac.cc
│       └── tests/
├── scripts/                    # Python utilities
│   ├── quad_rope_lift.py       # Single-quad 2D prototype
│   ├── generate_figures.py     # Paper figure generation
│   ├── generate_video.py       # Supplementary video generation
│   └── run_monte_carlo.sh      # Monte Carlo batch runner
└── outputs/
    └── logs/                   # Timestamped simulation data
        └── README.md           # Log file format documentation
```

## Quick Start

```bash
# Build
cd Research/cpp
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_PREFIX_PATH=/opt/drake
ninja

# Run (opens Meshcat at http://localhost:7000)
./quad_rope_lift

# Run with options
./quad_rope_lift --seed 42 --duration 50 --num-quads 3
./quad_rope_lift --headless  # No visualization (faster)
```

## Key Components

### C++ Simulation (`cpp/`)

The main simulation implements N quadcopters with bead-chain ropes lifting a payload. See [cpp/README.md](cpp/README.md) for detailed build instructions, architecture, and data logging documentation.

**GPAC Layers:**
| Layer | File | Rate | Purpose |
|-------|------|------|---------|
| 1 | `gpac_load_tracking_controller.h/cc` | 50 Hz | Position + S² anti-swing |
| 2 | `gpac_quadcopter_controller.h/cc` | 200 Hz | Geometric SO(3) attitude |
| 3 | `concurrent_learning_estimator.h/cc` | 200 Hz | Adaptive mass estimation |
| 4 | `extended_state_observer.h/cc` | 500 Hz | Disturbance estimation |
| Safety | `gpac_cbf_safety_filter.h/cc` | — | CBF constraint enforcement |

### Python Prototype (`scripts/`)

`quad_rope_lift.py` is a single-quadcopter 2D (x/z) prototype demonstrating slack-to-taut rope dynamics. See [scripts/README.md](scripts/README.md) for details.

### Post-Processing Scripts

- **`generate_figures.py`** — Generates all paper figures from simulation CSV logs
- **`generate_video.py`** — Creates supplementary video from simulation data
- **`run_monte_carlo.sh`** — Runs batch Monte Carlo simulations across multiple seeds

## Output

Simulation data is logged to timestamped CSV directories. See [outputs/logs/README.md](outputs/logs/README.md) for the complete file format specification.
