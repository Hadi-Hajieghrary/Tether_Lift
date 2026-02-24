#!/usr/bin/env python3
"""
Extract detailed per-seed and aggregate statistics from Monte Carlo runs.
Reads config.txt and CSV data from seed_001..seed_004, computes metrics,
and prints a structured summary to stdout.
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import OrderedDict

sys.path.insert(0, '/workspaces/Tether_Lift/Research/scripts')
from generate_figures import parse_config, load_csv, build_reference_trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_MIN_THRESHOLD = 2.0  # N — CBF minimum tension
T_CUTOFF = 6.0         # seconds — post-pickup steady flight
NUM_DRONES = 3
SEED_DIRS = [
    '/workspaces/Tether_Lift/outputs/monte_carlo/seed_001',
    '/workspaces/Tether_Lift/outputs/monte_carlo/seed_002',
    '/workspaces/Tether_Lift/outputs/monte_carlo/seed_003',
    '/workspaces/Tether_Lift/outputs/monte_carlo/seed_004',
]

# ---------------------------------------------------------------------------
# Helper: compute RMSE
# ---------------------------------------------------------------------------
def rmse(err):
    return np.sqrt(np.mean(err**2))

# ---------------------------------------------------------------------------
# Process one seed
# ---------------------------------------------------------------------------
def process_seed(seed_dir):
    """Return an OrderedDict of all metrics for one seed."""
    results = OrderedDict()
    seed_name = os.path.basename(seed_dir)

    # ---- 1. Config --------------------------------------------------------
    cfg = parse_config(seed_dir)
    results['random_seed'] = int(cfg.get('random_seed', -1))
    results['quad_0_rope_length'] = cfg.get('quad_0_rope_length_sampled', np.nan)
    results['quad_1_rope_length'] = cfg.get('quad_1_rope_length_sampled', np.nan)
    results['quad_2_rope_length'] = cfg.get('quad_2_rope_length_sampled', np.nan)
    results['avg_rope_length'] = cfg.get('avg_rope_length', np.nan)
    results['payload_mass'] = cfg.get('payload_mass', np.nan)
    results['formation_radius'] = cfg.get('formation_radius', np.nan)

    # ---- 2. Trajectories — tracking error ---------------------------------
    traj = load_csv(seed_dir, 'trajectories.csv')
    time = traj['time'].values
    ref = build_reference_trajectory(cfg, time)

    # Post-pickup mask
    mask = time > T_CUTOFF

    load_x = traj['load_x'].values
    load_y = traj['load_y'].values
    load_z = traj['load_z'].values

    ex = load_x - ref[:, 0]
    ey = load_y - ref[:, 1]
    ez = load_z - ref[:, 2]

    e3d = np.sqrt(ex**2 + ey**2 + ez**2)
    exy = np.sqrt(ex**2 + ey**2)

    results['load_3d_rmse_m'] = rmse(e3d[mask])
    results['load_3d_max_m'] = np.max(np.abs(e3d[mask]))
    results['load_xy_rmse_m'] = rmse(exy[mask])
    results['load_z_rmse_m'] = rmse(ez[mask])
    results['load_3d_rmse_cm'] = results['load_3d_rmse_m'] * 100
    results['load_3d_max_cm'] = results['load_3d_max_m'] * 100
    results['load_xy_rmse_cm'] = results['load_xy_rmse_m'] * 100
    results['load_z_rmse_cm'] = results['load_z_rmse_m'] * 100

    # ---- 3. Tensions -------------------------------------------------------
    tens = load_csv(seed_dir, 'tensions.csv')
    t_tens = tens['time'].values
    mask_t = t_tens > T_CUTOFF

    for i in range(NUM_DRONES):
        col = f'rope{i}_mag'
        vals = tens[col].values[mask_t]
        results[f'cable{i}_mean_N'] = np.mean(vals)
        results[f'cable{i}_std_N'] = np.std(vals)
        results[f'cable{i}_min_N'] = np.min(vals)
        results[f'cable{i}_max_N'] = np.max(vals)
        results[f'cable{i}_below_Tmin'] = int(np.sum(vals < T_MIN_THRESHOLD))

    # Aggregate tension stats across all cables
    all_tens = np.concatenate([tens[f'rope{i}_mag'].values[mask_t] for i in range(NUM_DRONES)])
    results['tension_overall_mean_N'] = np.mean(all_tens)
    results['tension_overall_min_N'] = np.min(all_tens)
    results['tension_overall_max_N'] = np.max(all_tens)
    results['tension_total_below_Tmin'] = sum(results[f'cable{i}_below_Tmin'] for i in range(NUM_DRONES))

    # ---- 4. Estimator outputs — ESKF and load estimator RMSE ---------------
    est = load_csv(seed_dir, 'estimator_outputs.csv')
    t_est = est['time'].values

    # Need to align estimator and trajectory timestamps
    # Both are sampled at same times in the sim output
    for i in range(NUM_DRONES):
        true_x = traj[f'drone{i}_x'].values
        true_y = traj[f'drone{i}_y'].values
        true_z = traj[f'drone{i}_z'].values
        est_x = est[f'drone{i}_est_x'].values
        est_y = est[f'drone{i}_est_y'].values
        est_z = est[f'drone{i}_est_z'].values

        # Handle possible length mismatch (use shorter)
        n = min(len(true_x), len(est_x))
        err_3d = np.sqrt(
            (true_x[:n] - est_x[:n])**2 +
            (true_y[:n] - est_y[:n])**2 +
            (true_z[:n] - est_z[:n])**2
        )
        # Use time from trajectory for the mask
        t_common = traj['time'].values[:n]
        mask_e = t_common > T_CUTOFF
        results[f'eskf_drone{i}_rmse_cm'] = rmse(err_3d[mask_e]) * 100

    # Load estimator
    n = min(len(traj), len(est))
    load_true = np.column_stack([traj['load_x'].values[:n],
                                  traj['load_y'].values[:n],
                                  traj['load_z'].values[:n]])
    load_est = np.column_stack([est['load_est_x'].values[:n],
                                 est['load_est_y'].values[:n],
                                 est['load_est_z'].values[:n]])
    load_err = np.sqrt(np.sum((load_true - load_est)**2, axis=1))
    t_common = traj['time'].values[:n]
    mask_le = t_common > T_CUTOFF
    results['load_est_rmse_cm'] = rmse(load_err[mask_le]) * 100

    # ---- 5. Attitude data — max tilt + cable angle -------------------------
    att_path = os.path.join(seed_dir, 'attitude_data.csv')
    if os.path.exists(att_path):
        att = load_csv(seed_dir, 'attitude_data.csv')
        t_att = att['time'].values
        mask_a = t_att > T_CUTOFF

        max_tilt = 0.0
        for i in range(NUM_DRONES):
            roll = att[f'drone{i}_roll'].values[mask_a]
            pitch = att[f'drone{i}_pitch'].values[mask_a]
            tilt = np.degrees(np.sqrt(roll**2 + pitch**2))
            max_tilt = max(max_tilt, np.max(tilt))
            results[f'drone{i}_max_tilt_deg'] = np.max(tilt)

        results['max_tilt_deg'] = max_tilt

        # Cable angle from tensions data
        max_cable_angle = 0.0
        for i in range(NUM_DRONES):
            mag = tens[f'rope{i}_mag'].values[mask_t]
            fz = tens[f'rope{i}_fz'].values[mask_t]
            with np.errstate(divide='ignore', invalid='ignore'):
                cos_theta = np.where(mag > 0.5, np.abs(fz) / mag, 1.0)
                cos_theta = np.clip(cos_theta, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_theta))
            max_cable_angle = max(max_cable_angle, np.max(angle_deg))
            results[f'cable{i}_max_angle_deg'] = np.max(angle_deg)

        results['max_cable_angle_deg'] = max_cable_angle
    else:
        results['max_tilt_deg'] = np.nan
        results['max_cable_angle_deg'] = np.nan

    # ---- 6. Control efforts — thrust per drone -----------------------------
    ctrl = load_csv(seed_dir, 'control_efforts.csv')
    t_ctrl = ctrl['time'].values
    mask_c = t_ctrl > T_CUTOFF

    for i in range(NUM_DRONES):
        fx = ctrl[f'drone{i}_f_x'].values[mask_c]
        fy = ctrl[f'drone{i}_f_y'].values[mask_c]
        fz = ctrl[f'drone{i}_f_z'].values[mask_c]
        thrust = np.sqrt(fx**2 + fy**2 + fz**2)
        results[f'drone{i}_mean_thrust_N'] = np.mean(thrust)
        results[f'drone{i}_max_thrust_N'] = np.max(thrust)

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    all_results = []

    print("=" * 90)
    print("MONTE CARLO STATISTICS — 4 Seeds")
    print("=" * 90)

    for seed_dir in SEED_DIRS:
        seed_name = os.path.basename(seed_dir)
        print(f"\n{'─' * 90}")
        print(f"  {seed_name}")
        print(f"{'─' * 90}")

        res = process_seed(seed_dir)
        all_results.append(res)

        # --- Print config ---
        print(f"\n  [Config]")
        print(f"    Random seed          : {res['random_seed']}")
        print(f"    Rope lengths (m)     : Q0={res['quad_0_rope_length']:.4f}  "
              f"Q1={res['quad_1_rope_length']:.4f}  "
              f"Q2={res['quad_2_rope_length']:.4f}")
        print(f"    Avg rope length (m)  : {res['avg_rope_length']:.4f}")
        print(f"    Payload mass (kg)    : {res['payload_mass']:.3f}")
        print(f"    Formation radius (m) : {res['formation_radius']:.3f}")

        # --- Print tracking ---
        print(f"\n  [Load Tracking Error — post-pickup t>{T_CUTOFF}s]")
        print(f"    3D RMSE              : {res['load_3d_rmse_cm']:.3f} cm")
        print(f"    3D Max Error         : {res['load_3d_max_cm']:.3f} cm")
        print(f"    XY (Horiz) RMSE      : {res['load_xy_rmse_cm']:.3f} cm")
        print(f"    Z  (Vert)  RMSE      : {res['load_z_rmse_cm']:.3f} cm")

        # --- Print tensions ---
        print(f"\n  [Cable Tensions — post-pickup t>{T_CUTOFF}s]")
        for i in range(NUM_DRONES):
            print(f"    Cable {i}: mean={res[f'cable{i}_mean_N']:.3f} N  "
                  f"std={res[f'cable{i}_std_N']:.3f} N  "
                  f"min={res[f'cable{i}_min_N']:.3f} N  "
                  f"max={res[f'cable{i}_max_N']:.3f} N  "
                  f"below T_min={res[f'cable{i}_below_Tmin']}")
        print(f"    Overall: mean={res['tension_overall_mean_N']:.3f} N  "
              f"min={res['tension_overall_min_N']:.3f} N  "
              f"max={res['tension_overall_max_N']:.3f} N  "
              f"total_below_Tmin={res['tension_total_below_Tmin']}")

        # --- Print estimator ---
        print(f"\n  [Estimator RMSE — post-pickup t>{T_CUTOFF}s]")
        for i in range(NUM_DRONES):
            print(f"    ESKF Drone {i} pos RMSE : {res[f'eskf_drone{i}_rmse_cm']:.3f} cm")
        print(f"    Load est. RMSE        : {res['load_est_rmse_cm']:.3f} cm")

        # --- Print attitude/cable angles ---
        if not np.isnan(res.get('max_tilt_deg', np.nan)):
            print(f"\n  [Attitude & Cable Angles — post-pickup t>{T_CUTOFF}s]")
            for i in range(NUM_DRONES):
                print(f"    Drone {i} max tilt     : {res[f'drone{i}_max_tilt_deg']:.2f} deg")
            print(f"    Max tilt (any drone)  : {res['max_tilt_deg']:.2f} deg")
            for i in range(NUM_DRONES):
                print(f"    Cable {i} max angle    : {res[f'cable{i}_max_angle_deg']:.2f} deg")
            print(f"    Max cable angle       : {res['max_cable_angle_deg']:.2f} deg")

        # --- Print control effort ---
        print(f"\n  [Control Effort — post-pickup t>{T_CUTOFF}s]")
        for i in range(NUM_DRONES):
            print(f"    Drone {i}: mean thrust={res[f'drone{i}_mean_thrust_N']:.3f} N  "
                  f"max thrust={res[f'drone{i}_max_thrust_N']:.3f} N")

    # =======================================================================
    # AGGREGATE STATISTICS
    # =======================================================================
    print(f"\n\n{'=' * 90}")
    print("AGGREGATE STATISTICS ACROSS 4 SEEDS")
    print(f"{'=' * 90}")

    # Define which metrics to aggregate and their display names/units
    aggregate_keys = [
        ('load_3d_rmse_cm',      'Load 3D RMSE',           'cm'),
        ('load_3d_max_cm',       'Load 3D Max Error',      'cm'),
        ('load_xy_rmse_cm',      'Load XY RMSE',           'cm'),
        ('load_z_rmse_cm',       'Load Z RMSE',            'cm'),
        ('tension_overall_mean_N', 'Tension Overall Mean', 'N'),
        ('tension_overall_min_N',  'Tension Overall Min',  'N'),
        ('tension_overall_max_N',  'Tension Overall Max',  'N'),
        ('tension_total_below_Tmin', 'Total Samples < T_min', 'count'),
    ]

    # Per-cable aggregates
    for i in range(NUM_DRONES):
        aggregate_keys.append((f'cable{i}_mean_N',  f'Cable {i} Mean Tension', 'N'))
        aggregate_keys.append((f'cable{i}_max_N',   f'Cable {i} Max Tension',  'N'))
        aggregate_keys.append((f'cable{i}_below_Tmin', f'Cable {i} Samples < T_min', 'count'))

    # Estimator
    for i in range(NUM_DRONES):
        aggregate_keys.append((f'eskf_drone{i}_rmse_cm', f'ESKF Drone {i} RMSE', 'cm'))
    aggregate_keys.append(('load_est_rmse_cm', 'Load Estimator RMSE', 'cm'))

    # Attitude
    aggregate_keys.append(('max_tilt_deg', 'Max Tilt Angle', 'deg'))
    aggregate_keys.append(('max_cable_angle_deg', 'Max Cable Angle', 'deg'))
    for i in range(NUM_DRONES):
        aggregate_keys.append((f'drone{i}_max_tilt_deg', f'Drone {i} Max Tilt', 'deg'))
        aggregate_keys.append((f'cable{i}_max_angle_deg', f'Cable {i} Max Angle', 'deg'))

    # Control
    for i in range(NUM_DRONES):
        aggregate_keys.append((f'drone{i}_mean_thrust_N', f'Drone {i} Mean Thrust', 'N'))
        aggregate_keys.append((f'drone{i}_max_thrust_N',  f'Drone {i} Max Thrust',  'N'))

    # Compute
    print(f"\n  {'Metric':<35s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}  {'Unit'}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*6}")

    aggregate_summary = OrderedDict()
    for key, label, unit in aggregate_keys:
        vals = np.array([r.get(key, np.nan) for r in all_results])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        m = np.mean(vals)
        s = np.std(vals)
        mn = np.min(vals)
        mx = np.max(vals)
        aggregate_summary[key] = {'mean': m, 'std': s, 'min': mn, 'max': mx}

        if unit == 'count':
            print(f"  {label:<35s} {m:>10.1f} {s:>10.1f} {mn:>10.0f} {mx:>10.0f}  {unit}")
        else:
            print(f"  {label:<35s} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}  {unit}")

    # =======================================================================
    # Compact per-seed table
    # =======================================================================
    print(f"\n\n{'=' * 90}")
    print("COMPACT PER-SEED TABLE")
    print(f"{'=' * 90}")

    compact_keys = [
        ('random_seed',          'Seed',           '',    'd'),
        ('quad_0_rope_length',   'L0 (m)',         'm',   '.4f'),
        ('quad_1_rope_length',   'L1 (m)',         'm',   '.4f'),
        ('quad_2_rope_length',   'L2 (m)',         'm',   '.4f'),
        ('payload_mass',         'M_L (kg)',       'kg',  '.2f'),
        ('load_3d_rmse_cm',      '3D RMSE (cm)',   'cm',  '.2f'),
        ('load_3d_max_cm',       '3D Max (cm)',    'cm',  '.2f'),
        ('load_xy_rmse_cm',      'XY RMSE (cm)',   'cm',  '.2f'),
        ('load_z_rmse_cm',       'Z RMSE (cm)',    'cm',  '.2f'),
        ('tension_overall_min_N','T_min (N)',      'N',   '.3f'),
        ('tension_overall_max_N','T_max (N)',      'N',   '.3f'),
        ('tension_total_below_Tmin', '#<T_min',    '',    'd'),
        ('load_est_rmse_cm',     'Load Est (cm)',  'cm',  '.2f'),
        ('max_tilt_deg',         'Max Tilt (deg)', 'deg', '.2f'),
        ('max_cable_angle_deg',  'Max Cable (deg)','deg', '.2f'),
    ]

    # Header
    hdr = f"  {'Metric':<20s}"
    for i, sd in enumerate(SEED_DIRS):
        hdr += f" {'Seed ' + str(i+1):>12s}"
    hdr += f" {'Mean':>12s} {'Std':>12s}"
    print(hdr)
    print(f"  {'─'*20}" + f" {'─'*12}" * (len(SEED_DIRS) + 2))

    for key, label, unit, fmt in compact_keys:
        row = f"  {label:<20s}"
        vals = []
        for r in all_results:
            v = r.get(key, np.nan)
            vals.append(v)
            if fmt == 'd':
                row += f" {int(v):>12d}"
            else:
                row += f" {v:>12{fmt}}"
        arr = np.array(vals)
        arr_clean = arr[~np.isnan(arr)]
        if fmt == 'd':
            row += f" {np.mean(arr_clean):>12.1f} {np.std(arr_clean):>12.1f}"
        else:
            row += f" {np.mean(arr_clean):>12{fmt}} {np.std(arr_clean):>12{fmt}}"
        print(row)

    print(f"\n{'=' * 90}")
    print("Done. All metrics computed for t > {:.1f}s (post-pickup steady flight).".format(T_CUTOFF))
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
