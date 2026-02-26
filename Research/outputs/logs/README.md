# Simulation Data Logs

This folder contains timestamped simulation data from the **Multi-Quadcopter Cooperative Payload Transport** simulation. Each simulation run creates a subfolder named with the timestamp `YYYYMMDD_HHMMSS`.

---

## Folder Structure

```
logs/
├── README.md                    # This file
└── YYYYMMDD_HHMMSS/            # Timestamped simulation run folder
    ├── config.txt              # Simulation configuration parameters
    ├── trajectories.csv        # Ground truth poses and velocities
    ├── imu_measurements.csv    # IMU sensor data (accelerometer + gyroscope)
    ├── barometer_measurements.csv  # Altitude sensor data
    ├── gps_measurements.csv    # GPS position measurements
    ├── estimator_outputs.csv   # State estimator outputs
    ├── tensions.csv            # Rope tension forces
    ├── control_efforts.csv     # Controller outputs (forces + torques)
    ├── attitude_data.csv       # Attitude angles and errors
    └── wind_disturbance.csv    # Wind velocity disturbances
```

---

## Log File Specifications

| File | Description | Sample Rate | Typical Size |
|------|-------------|-------------|--------------|
| `config.txt` | Simulation parameters | Once | ~2 KB |
| `trajectories.csv` | Ground truth state | 100 Hz | ~2.5 MB |
| `imu_measurements.csv` | 6-DOF IMU per drone | 100 Hz | ~800 KB |
| `barometer_measurements.csv` | Altitude per drone | 100 Hz | ~200 KB |
| `gps_measurements.csv` | GPS positions | 100 Hz | ~600 KB |
| `estimator_outputs.csv` | Estimated states | 100 Hz | ~1.2 MB |
| `tensions.csv` | Rope forces | 100 Hz | ~500 KB |
| `control_efforts.csv` | Control commands | 100 Hz | ~800 KB |
| `attitude_data.csv` | Attitude data | 100 Hz | ~1.5 MB |
| `wind_disturbance.csv` | Wind vectors | 100 Hz | ~200 KB |

**Total per simulation:** ~8 MB for 50 seconds @ 100 Hz

---

## Detailed File Descriptions

### 1. `config.txt` — Simulation Configuration

Contains all simulation parameters for reproducibility.

**Contents:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `num_quadcopters` | Number of drones in formation | 3 |
| `payload_mass` | Load mass [kg] | 3.0 |
| `quadcopter_mass` | Mass per drone [kg] | 1.5 |
| `formation_radius` | Circular formation radius [m] | 0.5 |
| `initial_altitude` | Drone starting height [m] | 1.2 |
| `avg_rope_length` | Mean rope length [m] | 1.0 |
| `quad_N_rope_length_mean` | Per-drone rope length mean [m] | varies |
| `quad_N_rope_length_sampled` | Actual sampled rope length [m] | varies |
| `num_rope_beads` | Bead-chain discretization | 8 |
| `segment_stiffness` | Rope spring constant [N/m] | 585.8 |
| `segment_damping` | Rope damping [N·s/m] | 21.0 |
| `simulation_duration` | Total sim time [s] | 50.0 |
| `simulation_time_step` | Physics timestep [s] | 0.0002 |
| `random_seed` | RNG seed for reproducibility | 42 |
| `waypoint_N_*` | Trajectory waypoint definitions | — |

---

### 2. `trajectories.csv` — Ground Truth State

**Description:** Complete ground truth poses and velocities for the load and all quadcopters. This is the "perfect" state from the physics simulation.

**Columns (for N=3 drones):**
```
time,
load_x, load_y, load_z,                    # Load position [m]
load_vx, load_vy, load_vz,                 # Load velocity [m/s]
load_qw, load_qx, load_qy, load_qz,        # Load orientation (quaternion)
load_wx, load_wy, load_wz,                 # Load angular velocity [rad/s]
drone0_x, drone0_y, drone0_z,              # Drone 0 position [m]
drone0_vx, drone0_vy, drone0_vz,           # Drone 0 velocity [m/s]
drone0_qw, drone0_qx, drone0_qy, drone0_qz,# Drone 0 orientation (quaternion)
drone0_wx, drone0_wy, drone0_wz,           # Drone 0 angular velocity [rad/s]
drone1_*, drone2_*, ...                    # Repeated for each drone
```

**Units:**
- Position: meters [m]
- Velocity: meters per second [m/s]
- Orientation: unit quaternion (w, x, y, z) — Hamilton convention
- Angular velocity: radians per second [rad/s]

**Use Cases:**
- Ground truth for estimator validation
- Trajectory tracking error analysis
- Animation/visualization playback

---

### 3. `imu_measurements.csv` — IMU Sensor Data

**Description:** Simulated 6-DOF Inertial Measurement Unit readings for each quadcopter. Includes realistic sensor noise and bias.

**Columns (for N=3 drones):**
```
time,
drone0_ax, drone0_ay, drone0_az,   # Drone 0 accelerometer [m/s²]
drone0_wx, drone0_wy, drone0_wz,   # Drone 0 gyroscope [rad/s]
drone1_ax, ..., drone1_wz,         # Drone 1
drone2_ax, ..., drone2_wz          # Drone 2
```

**Sensor Model:**
| Parameter | Accelerometer | Gyroscope |
|-----------|---------------|-----------|
| White noise density | 0.004 m/s²/√Hz | 0.0005 rad/s/√Hz |
| Bias instability | 1e-3 m/s² | 1e-4 rad/s |
| Bias time constant | 3600 s | 3600 s |
| Sample rate | 200 Hz (logged @ 100 Hz) | 200 Hz |

**Notes:**
- Accelerometer measures specific force (includes gravity in body frame)
- Gyroscope measures angular velocity in body frame
- Gauss-Markov bias model for realistic drift
- Each drone has independent random seed for uncorrelated noise

---

### 4. `barometer_measurements.csv` — Altitude Sensor

**Description:** Barometric altitude measurements for each quadcopter.

**Columns:**
```
time,
drone0_altitude,   # Drone 0 barometer reading [m]
drone1_altitude,   # Drone 1
drone2_altitude    # Drone 2
```

**Sensor Model:**
| Parameter | Value |
|-----------|-------|
| White noise σ | 0.3 m |
| Correlated noise σ | 0.2 m |
| Correlation time | 5.0 s |
| Bias drift rate | 0.002 m/s |
| Sample rate | 25 Hz (logged @ 100 Hz) |

**Notes:**
- Three-component noise model: white + correlated (exponentially decaying) + drift
- Provides vertical reference for altitude-hold and EKF fusion
- Sample-and-hold between sensor updates

---

### 5. `gps_measurements.csv` — GPS Position

**Description:** Simulated GPS position measurements for the load and all quadcopters.

**Columns:**
```
time,
load_gps_x, load_gps_y, load_gps_z,      # Load GPS [m]
drone0_gps_x, drone0_gps_y, drone0_gps_z, # Drone 0 GPS [m]
drone1_gps_x, drone1_gps_y, drone1_gps_z, # Drone 1 GPS [m]
drone2_gps_x, drone2_gps_y, drone2_gps_z  # Drone 2 GPS [m]
```

**Sensor Model:**
| Parameter | Value |
|-----------|-------|
| Horizontal noise σ (x, y) | 0.02 m |
| Vertical noise σ (z) | 0.05 m |
| Sample rate | 10 Hz (logged @ 100 Hz) |

**Notes:**
- Position noise follows HDOP/VDOP patterns (vertical worse than horizontal)
- Sample-and-hold between 10 Hz updates
- Used for position feedback and EKF measurement updates

---

### 6. `estimator_outputs.csv` — State Estimator

**Description:** Output of the state estimation system (EKF/ESKF) for the load and all drones.

**Columns:**
```
time,
load_est_x, load_est_y, load_est_z,           # Estimated load position [m]
load_est_vx, load_est_vy, load_est_vz,        # Estimated load velocity [m/s]
drone0_est_x, drone0_est_y, drone0_est_z,     # Estimated drone 0 position [m]
drone0_est_vx, drone0_est_vy, drone0_est_vz,  # Estimated drone 0 velocity [m/s]
drone1_est_*, drone2_est_*, ...               # Repeated for each drone
```

**Notes:**
- Compare with `trajectories.csv` to compute estimation error
- Fuses IMU, GPS, and barometer measurements
- 15-state ESKF (position, velocity, attitude, IMU biases)

---

### 7. `tensions.csv` — Rope Tension Forces

**Description:** Tension magnitude and force vectors for each rope connecting drones to the payload.

**Columns:**
```
time,
rope0_mag,                      # Rope 0 tension magnitude [N]
rope0_fx, rope0_fy, rope0_fz,   # Rope 0 force vector [N]
rope1_mag, rope1_fx, rope1_fy, rope1_fz,
rope2_mag, rope2_fx, rope2_fy, rope2_fz
```

**Notes:**
- Tension is zero when rope is slack (stretch ≤ 0)
- Force vector points from quadcopter toward payload
- Critical for analyzing pickup transients and load sharing
- Magnitude = ||[fx, fy, fz]||

**Typical Values:**
- Hover: ~(m_payload × g) / N per rope ≈ 9.8 N for 3kg load with 3 drones
- Dynamic: varies with trajectory acceleration

---

### 8. `control_efforts.csv` — Controller Outputs

**Description:** Control commands generated by the quadcopter controllers.

**Columns:**
```
time,
drone0_tau_x, drone0_tau_y, drone0_tau_z,  # Drone 0 body torques [N·m]
drone0_f_x, drone0_f_y, drone0_f_z,        # Drone 0 body forces [N]
drone1_tau_*, drone1_f_*,                   # Drone 1
drone2_tau_*, drone2_f_*                    # Drone 2
```

**Notes:**
- `tau_*`: Body-frame torques for roll/pitch/yaw control
- `f_z`: Thrust (typically ~22 N for hover = m×g = 1.5×9.81 + share of payload)
- `f_x, f_y`: Usually zero (thrust along body z-axis)
- Saturated within limits: thrust [0, 150] N, torque [-10, 10] N·m

---

### 9. `attitude_data.csv` — Attitude Information

**Description:** Euler angles, desired quaternions, and attitude tracking errors for each drone.

**Columns:**
```
time,
drone0_roll, drone0_pitch, drone0_yaw,           # Euler angles [rad]
drone0_des_qw, drone0_des_qx, drone0_des_qy, drone0_des_qz,  # Desired quaternion
drone0_err_x, drone0_err_y, drone0_err_z,        # Attitude error (SO(3)) [rad]
drone1_*, drone2_*, ...
```

**Notes:**
- Euler angles in ZYX (yaw-pitch-roll) convention
- Desired quaternion from geometric attitude controller
- Error: SO(3) rotation error e_R = ½(R_d^T R - R^T R_d)^∨

---

### 10. `wind_disturbance.csv` — Wind Velocities

**Description:** Wind disturbance velocity vector acting on the system.

**Columns:**
```
time,
wind_vx,   # Wind velocity x-component [m/s]
wind_vy,   # Wind velocity y-component [m/s]
wind_vz    # Wind velocity z-component [m/s]
```

**Wind Model (Dryden Turbulence):**
| Parameter | Value |
|-----------|-------|
| Mean wind | [1.0, 0.5, 0.0] m/s |
| Turbulence σ (u, v, w) | [0.5, 0.5, 0.25] m/s |
| Altitude scaling | Enabled |

**Notes:**
- Combines mean wind + turbulent fluctuations
- Dryden turbulence model with spatial correlation
- Altitude-dependent intensity scaling
- Applied as aerodynamic drag on quadcopters and payload

---

## Data Analysis Examples

### Python — Load Trajectory Tracking Error

```python
import pandas as pd
import numpy as np

# Load ground truth and estimates
traj = pd.read_csv('trajectories.csv')
est = pd.read_csv('estimator_outputs.csv')

# Compute position error
error_x = traj['load_x'] - est['load_est_x']
error_y = traj['load_y'] - est['load_est_y']
error_z = traj['load_z'] - est['load_est_z']
error_norm = np.sqrt(error_x**2 + error_y**2 + error_z**2)

print(f"Mean position error: {error_norm.mean():.4f} m")
print(f"Max position error: {error_norm.max():.4f} m")
```

### Python — Tension Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

tensions = pd.read_csv('tensions.csv')

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.plot(tensions['time'], tensions[f'rope{i}_mag'], label=f'Rope {i}')
plt.xlabel('Time [s]')
plt.ylabel('Tension [N]')
plt.title('Rope Tensions During Flight')
plt.legend()
plt.grid(True)
plt.show()
```

### MATLAB — IMU Noise Analysis

```matlab
data = readtable('imu_measurements.csv');

% Extract drone 0 accelerometer z-axis
az = data.drone0_az;
time = data.time;

% Compute Allan variance (simplified)
fs = 100;  % Sample rate
[avar, tau] = allanvar(az - mean(az), 'octave', fs);

loglog(tau, sqrt(avar))
xlabel('Averaging Time [s]')
ylabel('Allan Deviation [m/s²]')
title('Accelerometer Noise Characterization')
```

---

## Coordinate Frames

| Frame | Description | Convention |
|-------|-------------|------------|
| **World (W)** | Inertial frame | X=East, Y=North, Z=Up |
| **Body (B)** | Quadcopter body frame | X=Forward, Y=Left, Z=Up |
| **Load (L)** | Payload body frame | Attached at load center |

**Quaternion Convention:** Hamilton (w, x, y, z) where q = w + xi + yj + zk

---

## Notes

1. **Logging Rate:** All CSV files are sampled at 100 Hz regardless of internal sensor rates. Sensor data uses sample-and-hold between sensor updates.

2. **Time Synchronization:** All files share the same `time` column in seconds from simulation start.

3. **Buffer Flushing:** Data is flushed every 10 samples to prevent loss on early termination.

4. **Reproducibility:** Use the same `random_seed` in `config.txt` to reproduce identical noise sequences.

5. **Missing Data:** Zero values early in simulation may indicate sensors not yet initialized or ropes still slack.

---

## Contact

For questions about the logged data or simulation parameters, see the main project [README](../../README.md) or [C++ implementation documentation](../../cpp/README.md).
