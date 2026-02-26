# Scripts

## quad_rope_lift.py

This script simulates a single quadcopter lifting a load with a rope that can switch between **slack** and **taut** behavior.

It is designed to be a small, readable example that demonstrates:

- A **tension-only rope** model (no compression), so the rope can go slack.
- A simple **x/z motion plan** for the quad: **lift**, then move **left**, then move **right**.
- **Meshcat** visualization, including a rope line overlay and a quadcopter URDF for nicer visuals.

---

## What the simulation contains

### Bodies

- **Quad (dynamic body)**
  - Modeled as a single rigid body named `quad`.
  - Its motion is restricted to a vertical/horizontal plane using **two prismatic joints** (see below).
  - The URDF quadcopter model is welded onto this body for visualization.

- **Carriage (massless-ish intermediate body)**
  - A tiny-mass rigid body named `carriage`.
  - Exists only to create **two prismatic joints in series**:

    - `quad_x`: world → carriage (motion along +x)
    - `quad_z`: carriage → quad (motion along +z)

- **Load**
  - A free rigid body named `load` with a spherical collision shape.
  - Gravity is enabled by default in Drake’s `MultibodyPlant`.
  - The load collides with the ground.

- **Ground**
  - A HalfSpace collision geometry.

---

## Rope model (slack ↔ taut)

The rope is implemented by the `TensionOnlyRope` LeafSystem. It computes a force between:

- A body-fixed attachment point on the quad
- A body-fixed attachment point on the load

Let:

- `d` be the current distance between the two attachment points
- `L0` be the nominal (rest) rope length
- `e` be the unit direction from the load attachment point toward the quad attachment point
- `d_dot = eᵀ (v_quad - v_load)` be the rate of change of the distance

Then the rope tension is:

- If `d <= L0`: **slack**, tension `T = 0`
- If `d > L0`: **taut**, tension

  `T = k (d - L0) + c * max(d_dot, 0)`

Important details:

- The rope is **tension-only**: it never pushes.
- Damping is **pulling-only**: when the rope is shortening (`d_dot < 0`), damping does not apply.

The system applies equal-and-opposite spatial forces to the quad and the load.

---

## Controller and motion plan

The quad is controlled by `QuadXZController`.

### Inputs/outputs

- Input: the full plant state `x = [q; v]`
- Output: a 2D actuation vector `[u_x, u_z]` applied to the two prismatic joint actuators

### Control law (per axis)

Each axis uses PD control:

- `u_x = kp (x_des - x) + kd (xd_des - xd)`
- `u_z = kp (z_des - z) + kd (zd_des - zd) + u_ff_z`

`u_ff_z` is a constant feedforward term to help counter gravity.

### Time-based plan

The plan is parameterized by:

- `ramp_start`: when lifting begins
- `climb_rate`: vertical speed during the lift
- `x_left`, `x_right`: target x positions
- `move_pause`: pause time between phases
- `move_rate`: horizontal speed during moves

Sequence:

1. Hold at `(x0, z0)` until `ramp_start`.
2. Lift from `z0` to `z_final` at approximately `climb_rate`.
3. Pause.
4. Move from `x0` to `x_left` at approximately `move_rate`.
5. Pause.
6. Move from `x_left` to `x_right` at approximately `move_rate`.

The script computes a `sim_duration` long enough to complete all phases.

---

## Visualization (Meshcat)

### Quad URDF visuals

The script loads Drake’s quadrotor URDF and welds it to the dynamic `quad` body:

- `package://drake_models/skydio_2/quadrotor.urdf`

This provides a more realistic visualization while keeping the simulation dynamics simple.

### Rope line overlay

`RopeLineVisualizer` draws a Meshcat line segment between the two attachment points.

- It is colored **gray** when slack and **red** when taut (based on whether `d > L0`).
- Meshcat is cleared on startup to avoid stale visuals between runs.

---

## How to run

From the repository root:

- `/home/vscode/.venv/bin/python Research/scripts/quad_rope_lift.py`

The terminal will print a Meshcat URL (commonly `http://localhost:7000`). Open it in your browser to view the simulation.

---

## Common tuning knobs

All of these are defined near the top of `main()` in `quad_rope_lift.py`:

### Rope

- `rope_L0`: nominal rope length
- `rope_k`: stiffness (larger = closer to inextensible)
- `rope_c`: damping

### Motion

- `z_quad0`, `z_quad_final`: start and lift target height
- `x_left`, `x_right`: horizontal targets
- `climb_rate`, `move_rate`: speeds
- `move_pause`: pause between phases

### Control

- `kp`, `kd`: PD gains
- `u_max_x`, `u_max_z`: actuator saturation limits

If you see oscillations, start by reducing gains or increasing damping.

---

## Notes / gotchas

- The “ground deformable” warning from Drake can be ignored here; this example uses point contact.
- If the rope looks wrong when re-running, Meshcat state is cleared at startup; reloading the browser page also helps.
- The URDF model is used for visualization and also adds mass; the script updates `u_ff_z` to account for that additional mass.
