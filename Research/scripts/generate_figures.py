#!/usr/bin/env python3
"""
Generate camera-ready IEEE IROS 2026 figures from simulation outputs.

Usage:
    python generate_figures.py [--data-dir PATH] [--output-dir PATH] [--format pdf|png]

Reads CSV logs from the simulation and produces publication-quality figures
consistent with IEEE two-column formatting.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IEEE formatting constants
# ---------------------------------------------------------------------------
SINGLE_COL_WIDTH = 3.5        # inches (IEEE single column)
DOUBLE_COL_WIDTH = 7.16       # inches (IEEE double column)
FONT_SIZE = 8                 # IEEE standard caption/label size
FONT_SIZE_SMALL = 7
FONT_SIZE_TICK = 7
LINE_WIDTH = 0.8
LINE_WIDTH_THICK = 1.2
MARKER_SIZE = 3

# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    'drone0':   '#0072B2',  # blue
    'drone1':   '#D55E00',  # vermillion
    'drone2':   '#009E73',  # green
    'load':     '#CC79A7',  # pink
    'ref':      '#000000',  # black
    'limit':    '#E69F00',  # amber / warning
    'limit2':   '#56B4E9',  # sky blue
    'grey':     '#999999',
    'fill':     '#F0E442',  # yellow fill
}

DRONE_LABELS = ['Drone 0', 'Drone 1', 'Drone 2']
CABLE_LABELS = [
    r'Cable 0 ($L{=}0.914\,$m)',
    r'Cable 1 ($L{=}1.105\,$m)',
    r'Cable 2 ($L{=}0.995\,$m)',
]
DRONE_COLORS = [COLORS['drone0'], COLORS['drone1'], COLORS['drone2']]

# Flight phase boundaries (seconds) for shading / annotation
PHASES = [
    (0,  2,  'Pre-lift'),
    (2,  6,  'Pickup /\nAscent'),
    (7,  20, 'Fig-8 R'),
    (21, 36, 'Fig-8 L'),
    (39, 43, 'Descent'),
    (43, 50, 'Post'),
]


def setup_matplotlib():
    """Configure matplotlib for IEEE publication style."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_SMALL,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': LINE_WIDTH,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.handlelength': 1.5,
        'text.usetex': False,        # set True if full LaTeX available
        'mathtext.fontset': 'cm',    # Computer Modern math
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(data_dir, filename):
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path)


def parse_config(data_dir):
    """Parse config.txt into a dict."""
    cfg = {}
    path = os.path.join(data_dir, 'config.txt')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            # try numeric
            try:
                val = float(val)
            except ValueError:
                # try tuple
                m = re.match(r'\((.+)\)', val)
                if m:
                    val = tuple(float(x) for x in m.group(1).split(','))
            cfg[key] = val
        return cfg


def build_reference_trajectory(cfg, time):
    """Reconstruct minimum-jerk reference from waypoints for the load."""
    # Collect waypoints sorted by index
    wps = {}
    for k, v in cfg.items():
        m = re.match(r'waypoint_(\d+)_(.+)', k)
        if m:
            idx = int(m.group(1))
            field = m.group(2)
            wps.setdefault(idx, {})[field] = v
    # Sort by index
    wp_list = [wps[i] for i in sorted(wps.keys())]

    # Build segment endpoints
    segments = []
    for wp in wp_list:
        t_arrive = wp['arrival_time']
        t_hold = wp['hold_time']
        pos = np.array(wp['position'])
        segments.append((t_arrive, t_hold, pos))

    # Compute load offset: quadrotor centroid waypoints -> load position
    avg_rope = cfg.get('avg_rope_length', 1.0)
    payload_radius = cfg.get('payload_radius', 0.15)
    z_attach = 0.10
    eps_max = 0.15
    delta_z = z_attach + avg_rope * (1 + eps_max) + payload_radius
    load_offset = np.array([0, 0, delta_z])

    # Build time-position pairs for interpolation
    knots_t = []
    knots_p = []
    for t_arr, t_hold, pos in segments:
        p_load = pos - load_offset
        p_load[2] = max(p_load[2], payload_radius + 0.01)
        if knots_t and t_arr <= knots_t[-1]:
            t_arr = knots_t[-1] + 0.001
        knots_t.append(t_arr)
        knots_p.append(p_load)
        if t_hold > 0:
            knots_t.append(t_arr + t_hold)
            knots_p.append(p_load)

    knots_t = np.array(knots_t)
    knots_p = np.array(knots_p)

    # Minimum-jerk interpolation between consecutive knots
    def min_jerk_segment(t, t0, t1, p0, p1):
        tau = np.clip((t - t0) / (t1 - t0 + 1e-12), 0, 1)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        return p0 + (p1 - p0) * s[:, None] if hasattr(t, '__len__') else p0 + (p1 - p0) * s

    ref = np.zeros((len(time), 3))
    for i in range(len(time)):
        t = time[i]
        if t <= knots_t[0]:
            ref[i] = knots_p[0]
        elif t >= knots_t[-1]:
            ref[i] = knots_p[-1]
        else:
            # find segment
            seg_idx = np.searchsorted(knots_t, t, side='right') - 1
            seg_idx = min(seg_idx, len(knots_t) - 2)
            t0, t1 = knots_t[seg_idx], knots_t[seg_idx + 1]
            p0, p1 = knots_p[seg_idx], knots_p[seg_idx + 1]
            tau = np.clip((t - t0) / (t1 - t0 + 1e-12), 0, 1)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ref[i] = p0 + (p1 - p0) * s

    return ref


def add_phase_shading(ax, alpha=0.06):
    """Add light background shading for flight phases."""
    shade_colors = ['#DDDDDD', '#E8E8FF', '#FFE8E8', '#E8FFE8', '#FFF0DD', '#F0F0F0']
    ylim = ax.get_ylim()
    for i, (t0, t1, label) in enumerate(PHASES):
        ax.axvspan(t0, t1, alpha=alpha, color=shade_colors[i % len(shade_colors)],
                   zorder=0)
    ax.set_ylim(ylim)


def add_phase_labels(ax, y_frac=0.95, fontsize=5.5):
    """Add small phase labels at the top of the plot."""
    ylim = ax.get_ylim()
    y_pos = ylim[0] + y_frac * (ylim[1] - ylim[0])
    for t0, t1, label in PHASES:
        ax.text((t0 + t1) / 2, y_pos, label, ha='center', va='top',
                fontsize=fontsize, color='#666666', style='italic')


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_trajectory_3d(traj, cfg, ref, output_dir, fmt):
    """Fig 1: 3D trajectory — reference vs actual load + drone paths."""
    fig = plt.figure(figsize=(SINGLE_COL_WIDTH, 2.8))
    ax = fig.add_subplot(111, projection='3d')

    t = traj['time'].values

    # Reference
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
            color=COLORS['ref'], ls='--', lw=LINE_WIDTH, label='Reference', zorder=5)

    # Actual load
    ax.plot(traj['load_x'], traj['load_y'], traj['load_z'],
            color=COLORS['load'], lw=LINE_WIDTH_THICK, label='Payload', zorder=4)

    # Drones (thinner)
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        ax.plot(traj[f'drone{i}_x'], traj[f'drone{i}_y'], traj[f'drone{i}_z'],
                color=c, lw=0.4, alpha=0.6, label=lab)

    # Start / end markers
    ax.scatter(*ref[0], marker='o', s=20, color='green', zorder=10, label='Start')
    ax.scatter(*ref[-1], marker='s', s=20, color='red', zorder=10, label='End')

    ax.set_xlabel('X (m)', labelpad=1)
    ax.set_ylabel('Y (m)', labelpad=1)
    ax.set_zlabel('Z (m)', labelpad=1)
    ax.tick_params(axis='both', pad=1, labelsize=6)
    ax.view_init(elev=25, azim=-55)
    ax.legend(fontsize=5.5, loc='upper left', ncol=2, handlelength=1.2,
              columnspacing=0.8, borderpad=0.3, labelspacing=0.3)

    fig.savefig(os.path.join(output_dir, f'fig_trajectory_3d.{fmt}'))
    plt.close(fig)
    print('  [1/9] fig_trajectory_3d')


def fig_trajectory_2d(traj, cfg, ref, output_dir, fmt):
    """Fig 2: Top-down XY + altitude profile (two subplots)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 3.2),
                                    gridspec_kw={'height_ratios': [2, 1]})
    t = traj['time'].values

    # --- XY top-down ---
    ax1.plot(ref[:, 0], ref[:, 1], color=COLORS['ref'], ls='--', lw=LINE_WIDTH,
             label='Reference')
    ax1.plot(traj['load_x'], traj['load_y'], color=COLORS['load'],
             lw=LINE_WIDTH_THICK, label='Payload')
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        ax1.plot(traj[f'drone{i}_x'], traj[f'drone{i}_y'],
                 color=c, lw=0.4, alpha=0.5, label=lab)
    # Waypoint markers
    wps_xy = []
    for k, v in cfg.items():
        if k.endswith('_position') and 'waypoint' in k:
            wps_xy.append(v)
    if wps_xy:
        wps_xy = np.array(wps_xy)
        avg_rope = cfg.get('avg_rope_length', 1.0)
        ax1.scatter(wps_xy[:, 0], wps_xy[:, 1], marker='x', s=12,
                    color=COLORS['grey'], zorder=6, linewidths=0.6)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.legend(fontsize=5.5, ncol=2, loc='lower right', handlelength=1.2,
               columnspacing=0.8)
    ax1.grid(True)

    # --- Altitude profile ---
    ax2.plot(t, ref[:, 2], color=COLORS['ref'], ls='--', lw=LINE_WIDTH,
             label='Reference')
    ax2.plot(t, traj['load_z'], color=COLORS['load'], lw=LINE_WIDTH_THICK,
             label='Payload')
    for i, c in enumerate(DRONE_COLORS):
        ax2.plot(t, traj[f'drone{i}_z'], color=c, lw=0.4, alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_xlim(0, 50)
    ax2.legend(fontsize=5.5, loc='lower right', handlelength=1.2)
    ax2.grid(True)
    add_phase_shading(ax2)

    fig.tight_layout(h_pad=0.5)
    fig.savefig(os.path.join(output_dir, f'fig_trajectory_2d.{fmt}'))
    plt.close(fig)
    print('  [2/9] fig_trajectory_2d')


def fig_tracking_error(traj, ref, output_dir, fmt):
    """Fig 3: Position tracking error — per-axis + 3D norm."""
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 2.8), sharex=True)
    t = traj['time'].values

    ex = traj['load_x'].values - ref[:, 0]
    ey = traj['load_y'].values - ref[:, 1]
    ez = traj['load_z'].values - ref[:, 2]
    e3d = np.sqrt(ex**2 + ey**2 + ez**2)

    # Per-axis
    ax = axes[0]
    ax.plot(t, ex * 100, color=COLORS['drone0'], lw=LINE_WIDTH, label='$e_x$')
    ax.plot(t, ey * 100, color=COLORS['drone1'], lw=LINE_WIDTH, label='$e_y$')
    ax.plot(t, ez * 100, color=COLORS['drone2'], lw=LINE_WIDTH, label='$e_z$')
    ax.set_ylabel('Error (cm)')
    ax.legend(ncol=3, loc='upper right', handlelength=1.2, columnspacing=0.8)
    ax.grid(True)
    add_phase_shading(ax)

    # 3D norm
    ax = axes[1]
    ax.fill_between(t, 0, e3d * 100, alpha=0.15, color=COLORS['load'])
    ax.plot(t, e3d * 100, color=COLORS['load'], lw=LINE_WIDTH_THICK,
            label='$\\|e\\|_{3D}$')

    # RMSE annotation
    # Compute per-phase RMSE
    mask_fig8 = (t >= 7) & (t <= 36)
    rmse_all = np.sqrt(np.mean(e3d**2)) * 100
    rmse_fig8 = np.sqrt(np.mean(e3d[mask_fig8]**2)) * 100
    ax.axhline(rmse_all, color=COLORS['grey'], ls=':', lw=0.6)
    ax.text(48, rmse_all + 1.5, f'RMSE={rmse_all:.1f} cm',
            fontsize=FONT_SIZE_SMALL, ha='right', color=COLORS['grey'])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('3D Error (cm)')
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', handlelength=1.2)
    ax.grid(True)
    add_phase_shading(ax)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, f'fig_tracking_error.{fmt}'))
    plt.close(fig)
    print('  [3/9] fig_tracking_error')


def fig_cable_tensions(tensions, output_dir, fmt):
    """Fig 4: Cable tension profiles with CBF limits."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.0))
    t = tensions['time'].values

    for i, (c, lab) in enumerate(zip(DRONE_COLORS, CABLE_LABELS)):
        ax.plot(t, tensions[f'rope{i}_mag'], color=c, lw=LINE_WIDTH, label=lab)

    # CBF limits
    ax.axhline(2.0, color=COLORS['limit'], ls='--', lw=0.6,
               label='$T_{\\min}=2\\,$N')
    ax.axhline(60.0, color=COLORS['limit'], ls=':', lw=0.6,
               label='$T_{\\max}=60\\,$N')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tension (N)')
    ax.set_xlim(0, 50)
    ax.set_ylim(-1, 25)
    ax.legend(fontsize=5.5, ncol=2, loc='upper right', handlelength=1.2,
              columnspacing=0.6)
    ax.grid(True)
    add_phase_shading(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig_cable_tensions.{fmt}'))
    plt.close(fig)
    print('  [4/9] fig_cable_tensions')


def fig_estimation_error(traj, est, ref, output_dir, fmt):
    """Fig 5: Estimation accuracy — ESKF (drone) and load estimator errors."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 2.8), sharex=True)
    t = traj['time'].values

    # --- Drone ESKF errors ---
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        true_x = traj[f'drone{i}_x'].values
        true_y = traj[f'drone{i}_y'].values
        true_z = traj[f'drone{i}_z'].values
        est_x = est[f'drone{i}_est_x'].values
        est_y = est[f'drone{i}_est_y'].values
        est_z = est[f'drone{i}_est_z'].values
        err = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2 + (true_z - est_z)**2)
        ax1.plot(t, err * 100, color=c, lw=LINE_WIDTH, label=lab)

    ax1.set_ylabel('ESKF Error (cm)')
    ax1.legend(fontsize=5.5, ncol=3, loc='upper right', handlelength=1.2,
               columnspacing=0.8)
    ax1.grid(True)
    ax1.set_ylim(bottom=0)
    add_phase_shading(ax1)

    # --- Load estimator error ---
    load_true = traj[['load_x', 'load_y', 'load_z']].values
    load_est = est[['load_est_x', 'load_est_y', 'load_est_z']].values
    load_err = np.sqrt(np.sum((load_true - load_est)**2, axis=1))

    ax2.fill_between(t, 0, load_err * 100, alpha=0.15, color=COLORS['load'])
    ax2.plot(t, load_err * 100, color=COLORS['load'], lw=LINE_WIDTH_THICK,
             label='Decentral. load est.')

    rmse_load = np.sqrt(np.mean(load_err**2)) * 100
    ax2.axhline(rmse_load, color=COLORS['grey'], ls=':', lw=0.6)
    ax2.text(48, rmse_load + 2, f'RMSE={rmse_load:.1f} cm',
             fontsize=FONT_SIZE_SMALL, ha='right', color=COLORS['grey'])

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Load Est. Error (cm)')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=5.5, loc='upper right', handlelength=1.2)
    ax2.grid(True)
    add_phase_shading(ax2)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, f'fig_estimation_error.{fmt}'))
    plt.close(fig)
    print('  [5/9] fig_estimation_error')


def fig_safety_constraints(tensions, attitude, output_dir, fmt):
    """Fig 6: Safety constraint satisfaction — cable angle + tilt."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 2.8), sharex=True)
    t = tensions['time'].values

    # --- Cable angle from vertical ---
    # angle_i = arccos(-fz / mag)  (fz points downward from drone toward load)
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        mag = tensions[f'rope{i}_mag'].values
        fz = tensions[f'rope{i}_fz'].values
        # Cable direction: force on drone from cable, so direction is toward load
        # cos(theta) = |fz| / mag (fz is typically negative = downward)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = np.where(mag > 0.5, np.abs(fz) / mag, 1.0)
            cos_theta = np.clip(cos_theta, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_theta))
        ax1.plot(t, angle_deg, color=c, lw=LINE_WIDTH, label=lab)

    # CBF limit
    theta_max = np.degrees(0.6)  # 34.4 deg
    ax1.axhline(theta_max, color=COLORS['limit'], ls='--', lw=0.8,
                label=f'$\\theta_{{\\max}}={theta_max:.1f}°$')
    ax1.set_ylabel('Cable Angle (°)')
    ax1.legend(fontsize=5.5, ncol=2, loc='upper right', handlelength=1.2,
               columnspacing=0.6)
    ax1.grid(True)
    ax1.set_ylim(0, 55)
    add_phase_shading(ax1)

    # --- Quadrotor tilt ---
    t_att = attitude['time'].values
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        roll = attitude[f'drone{i}_roll'].values
        pitch = attitude[f'drone{i}_pitch'].values
        # Tilt = arccos(cos(roll)*cos(pitch)) for small angles ≈ sqrt(roll^2+pitch^2)
        tilt = np.sqrt(roll**2 + pitch**2)
        tilt_deg = np.degrees(tilt)
        ax2.plot(t_att, tilt_deg, color=c, lw=LINE_WIDTH, label=lab)

    phi_max = np.degrees(0.5)  # 28.6 deg
    ax2.axhline(phi_max, color=COLORS['limit'], ls='--', lw=0.8,
                label=f'$\\phi_{{\\max}}={phi_max:.1f}°$')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tilt Angle (°)')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 35)
    ax2.legend(fontsize=5.5, ncol=2, loc='upper right', handlelength=1.2,
               columnspacing=0.6)
    ax2.grid(True)
    add_phase_shading(ax2)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, f'fig_safety_constraints.{fmt}'))
    plt.close(fig)
    print('  [6/9] fig_safety_constraints')


def fig_control_effort(ctrl, output_dir, fmt):
    """Fig 7: Control effort — thrust magnitude + torque RMS per drone."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 2.8), sharex=True)
    t = ctrl['time'].values

    # --- Thrust magnitude ---
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        fx = ctrl[f'drone{i}_f_x'].values
        fy = ctrl[f'drone{i}_f_y'].values
        fz = ctrl[f'drone{i}_f_z'].values
        thrust = np.sqrt(fx**2 + fy**2 + fz**2)
        ax1.plot(t, thrust, color=c, lw=LINE_WIDTH, label=lab)

    # Hover thrust reference
    hover_thrust = (1.5 + 3.0 / 3) * 9.81  # m_Q*g + m_L*g/N
    ax1.axhline(hover_thrust, color=COLORS['grey'], ls=':', lw=0.6,
                label=f'Hover = {hover_thrust:.1f} N')
    ax1.set_ylabel('Thrust (N)')
    ax1.legend(fontsize=5.5, ncol=2, loc='upper right', handlelength=1.2,
               columnspacing=0.6)
    ax1.grid(True)
    add_phase_shading(ax1)

    # --- Torque magnitude ---
    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        tx = ctrl[f'drone{i}_tau_x'].values
        ty = ctrl[f'drone{i}_tau_y'].values
        tz = ctrl[f'drone{i}_tau_z'].values
        torque = np.sqrt(tx**2 + ty**2 + tz**2)
        ax2.plot(t, torque, color=c, lw=LINE_WIDTH, label=lab)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Torque (N$\\cdot$m)')
    ax2.set_xlim(0, 50)
    ax2.legend(fontsize=5.5, ncol=3, loc='upper right', handlelength=1.2,
               columnspacing=0.6)
    ax2.grid(True)
    add_phase_shading(ax2)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, f'fig_control_effort.{fmt}'))
    plt.close(fig)
    print('  [7/9] fig_control_effort')


def fig_wind_disturbance(wind, output_dir, fmt):
    """Fig 8: Wind disturbance velocity components."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 1.6))
    t = wind['time'].values

    ax.plot(t, wind['wind_vx'], color=COLORS['drone0'], lw=LINE_WIDTH,
            label='$v_{w,x}$')
    ax.plot(t, wind['wind_vy'], color=COLORS['drone1'], lw=LINE_WIDTH,
            label='$v_{w,y}$')
    ax.plot(t, wind['wind_vz'], color=COLORS['drone2'], lw=LINE_WIDTH,
            label='$v_{w,z}$')

    # Mean wind reference
    ax.axhline(1.0, color=COLORS['drone0'], ls=':', lw=0.4, alpha=0.5)
    ax.axhline(0.5, color=COLORS['drone1'], ls=':', lw=0.4, alpha=0.5)

    mag = np.sqrt(wind['wind_vx']**2 + wind['wind_vy']**2 + wind['wind_vz']**2)
    ax.fill_between(t, 0, mag, alpha=0.08, color=COLORS['grey'])
    ax.plot(t, mag, color=COLORS['grey'], lw=0.5, ls='--', label='$\\|v_w\\|$')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wind Vel. (m/s)')
    ax.set_xlim(0, 50)
    ax.legend(fontsize=5.5, ncol=4, loc='upper right', handlelength=1.2,
              columnspacing=0.6)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig_wind_disturbance.{fmt}'))
    plt.close(fig)
    print('  [8/9] fig_wind_disturbance')


def fig_attitude_tracking(attitude, output_dir, fmt):
    """Fig 9: Attitude errors (roll/pitch) for all drones."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 2.4), sharex=True)
    t = attitude['time'].values

    for i, (c, lab) in enumerate(zip(DRONE_COLORS, DRONE_LABELS)):
        roll_deg = np.degrees(attitude[f'drone{i}_roll'].values)
        pitch_deg = np.degrees(attitude[f'drone{i}_pitch'].values)
        ax1.plot(t, roll_deg, color=c, lw=LINE_WIDTH, label=lab)
        ax2.plot(t, pitch_deg, color=c, lw=LINE_WIDTH, label=lab)

    ax1.set_ylabel('Roll (°)')
    ax1.legend(fontsize=5.5, ncol=3, loc='upper right', handlelength=1.2,
               columnspacing=0.8)
    ax1.grid(True)
    add_phase_shading(ax1)

    ax2.set_ylabel('Pitch (°)')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim(0, 50)
    ax2.legend(fontsize=5.5, ncol=3, loc='upper right', handlelength=1.2,
               columnspacing=0.8)
    ax2.grid(True)
    add_phase_shading(ax2)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(output_dir, f'fig_attitude_tracking.{fmt}'))
    plt.close(fig)
    print('  [9/9] fig_attitude_tracking')


# ---------------------------------------------------------------------------
# Summary statistics (printed to stdout and saved as LaTeX table fragment)
# ---------------------------------------------------------------------------

def print_summary_stats(traj, ref, tensions, est, ctrl, wind, output_dir):
    """Compute and print summary statistics matching the paper tables."""
    t = traj['time'].values

    print('\n' + '=' * 60)
    print('SUMMARY STATISTICS')
    print('=' * 60)

    # --- Tracking error ---
    ex = (traj['load_x'].values - ref[:, 0]) * 100
    ey = (traj['load_y'].values - ref[:, 1]) * 100
    ez = (traj['load_z'].values - ref[:, 2]) * 100
    exy = np.sqrt(ex**2 + ey**2)
    e3d = np.sqrt(ex**2 + ey**2 + ez**2)

    phase_ranges = [
        ('Ascent (2-6s)', 2, 6),
        ('Fig-8 R (7-20s)', 7, 20),
        ('Fig-8 L (24-36s)', 24, 36),
        ('Descent (39-43s)', 39, 43),
        ('Post (43-50s)', 43, 50),
        ('Overall', 0, 50),
    ]

    print('\nPayload Tracking Error (cm):')
    print(f'  {"Phase":<22s} {"XY RMSE":>8s} {"XY Max":>8s} {"Z RMSE":>8s} '
          f'{"Z Max":>8s} {"3D RMSE":>8s} {"3D Max":>8s}')
    print('  ' + '-' * 62)
    for name, t0, t1 in phase_ranges:
        m = (t >= t0) & (t <= t1)
        print(f'  {name:<22s} {np.sqrt(np.mean(exy[m]**2)):8.1f} {np.max(np.abs(exy[m])):8.1f} '
              f'{np.sqrt(np.mean(ez[m]**2)):8.1f} {np.max(np.abs(ez[m])):8.1f} '
              f'{np.sqrt(np.mean(e3d[m]**2)):8.1f} {np.max(e3d[m]):8.1f}')

    # --- Cable tensions ---
    print('\nCable Tension Stats (N) during steady flight (6-38s):')
    m = (t >= 6) & (t <= 38)
    sampled_lengths = [0.914, 1.105, 0.995]
    print(f'  {"Cable":<10s} {"L (m)":>8s} {"Mean":>8s} {"Std":>8s} {"Min":>8s} {"Max":>8s}')
    print('  ' + '-' * 50)
    for i in range(3):
        mag = tensions[f'rope{i}_mag'].values[m]
        print(f'  Cable {i:<5d} {sampled_lengths[i]:8.3f} {np.mean(mag):8.2f} '
              f'{np.std(mag):8.2f} {np.min(mag):8.2f} {np.max(mag):8.2f}')
    total = sum(tensions[f'rope{i}_mag'].values[m] for i in range(3))
    print(f'  {"Total":<10s} {"---":>8s} {np.mean(total):8.2f} {np.std(total):8.2f} '
          f'{np.min(total):8.2f} {np.max(total):8.2f}')

    # --- Estimation errors ---
    print('\nEstimation Errors (cm):')
    for i in range(3):
        true_pos = traj[[f'drone{i}_x', f'drone{i}_y', f'drone{i}_z']].values
        est_pos = est[[f'drone{i}_est_x', f'drone{i}_est_y', f'drone{i}_est_z']].values
        err = np.sqrt(np.sum((true_pos - est_pos)**2, axis=1)) * 100
        print(f'  Drone {i} ESKF:  RMSE={np.sqrt(np.mean(err**2)):6.2f}  '
              f'Max={np.max(err):6.2f}')

    load_true = traj[['load_x', 'load_y', 'load_z']].values
    load_est = est[['load_est_x', 'load_est_y', 'load_est_z']].values
    load_err = np.sqrt(np.sum((load_true - load_est)**2, axis=1)) * 100
    print(f'  Load (decentr.): RMSE={np.sqrt(np.mean(load_err**2)):6.2f}  '
          f'Max={np.max(load_err):6.2f}')

    # --- Control effort ---
    print('\nControl Effort Stats:')
    for i in range(3):
        fx = ctrl[f'drone{i}_f_x'].values
        fy = ctrl[f'drone{i}_f_y'].values
        fz = ctrl[f'drone{i}_f_z'].values
        thrust = np.sqrt(fx**2 + fy**2 + fz**2)
        tx = ctrl[f'drone{i}_tau_x'].values
        ty = ctrl[f'drone{i}_tau_y'].values
        tz = ctrl[f'drone{i}_tau_z'].values
        torque = np.sqrt(tx**2 + ty**2 + tz**2)
        print(f'  Drone {i}: Thrust mean={np.mean(thrust):6.2f} N  max={np.max(thrust):6.2f} N  '
              f'Torque RMS={np.sqrt(np.mean(torque**2)):5.3f} Nm')

    # --- Wind ---
    print('\nWind Stats:')
    wmag = np.sqrt(wind['wind_vx']**2 + wind['wind_vy']**2 + wind['wind_vz']**2)
    print(f'  Mean vel: ({wind["wind_vx"].mean():.2f}, {wind["wind_vy"].mean():.2f}, '
          f'{wind["wind_vz"].mean():.2f}) m/s')
    print(f'  Magnitude: mean={wmag.mean():.2f}  std={wmag.std():.2f}  '
          f'range=[{wmag.min():.2f}, {wmag.max():.2f}] m/s')

    # --- Safety constraints ---
    print('\nSafety Constraints:')
    for i in range(3):
        mag = tensions[f'rope{i}_mag'].values
        fz = tensions[f'rope{i}_fz'].values
        valid = mag > 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_th = np.where(valid, np.abs(fz) / mag, 1.0)
            cos_th = np.clip(cos_th, -1, 1)
        angle = np.degrees(np.arccos(cos_th))
        print(f'  Cable {i} max angle: {np.max(angle[valid]):.1f}°  '
              f'(limit: 34.4°)')

    for i in range(3):
        roll = attitude_data[f'drone{i}_roll'].values
        pitch = attitude_data[f'drone{i}_pitch'].values
        tilt = np.degrees(np.sqrt(roll**2 + pitch**2))
        print(f'  Drone {i} max tilt: {np.max(tilt):.1f}°  '
              f'(limit: 28.6°)')

    print('=' * 60)

    # Save summary to text file
    # (capture above would be ideal, but let's write key numbers)
    summary_path = os.path.join(output_dir, 'summary_stats.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Payload 3D RMSE: {np.sqrt(np.mean(e3d**2)):.1f} cm\n')
        f.write(f'Payload 3D Max:  {np.max(e3d):.1f} cm\n')
        f.write(f'Load Est. RMSE:  {np.sqrt(np.mean(load_err**2)):.1f} cm\n')
        for i in range(3):
            true_pos = traj[[f'drone{i}_x', f'drone{i}_y', f'drone{i}_z']].values
            est_pos = est[[f'drone{i}_est_x', f'drone{i}_est_y', f'drone{i}_est_z']].values
            err = np.sqrt(np.sum((true_pos - est_pos)**2, axis=1)) * 100
            f.write(f'Drone {i} ESKF RMSE: {np.sqrt(np.mean(err**2)):.2f} cm\n')
        f.write(f'Wind mean magnitude: {wmag.mean():.2f} m/s\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate IEEE IROS 2026 figures')
    parser.add_argument('--data-dir', type=str,
                        default=None,
                        help='Path to simulation output directory')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Path to save figures')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'eps', 'svg'],
                        help='Output figure format (default: pdf)')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.data_dir is None:
        # Auto-detect: find most recent log directory
        log_base = repo_root.parent / 'Research' / 'outputs' / 'logs'
        if not log_base.exists():
            log_base = repo_root / 'outputs' / 'logs'
        if log_base.exists():
            log_dirs = sorted([d for d in log_base.iterdir() if d.is_dir()])
            if log_dirs:
                args.data_dir = str(log_dirs[-1])
    if args.data_dir is None:
        print('ERROR: Could not auto-detect data directory. Use --data-dir.', file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(repo_root / 'figures')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Data directory:   {args.data_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Format:           {args.format}')
    print()

    # Setup matplotlib
    setup_matplotlib()

    # Load data
    print('Loading data...')
    cfg = parse_config(args.data_dir)
    traj = load_csv(args.data_dir, 'trajectories.csv')
    tensions = load_csv(args.data_dir, 'tensions.csv')
    ctrl = load_csv(args.data_dir, 'control_efforts.csv')
    est = load_csv(args.data_dir, 'estimator_outputs.csv')
    global attitude_data
    attitude_data = load_csv(args.data_dir, 'attitude_data.csv')
    wind = load_csv(args.data_dir, 'wind_disturbance.csv')

    # Build reference trajectory
    print('Building reference trajectory from waypoints...')
    ref = build_reference_trajectory(cfg, traj['time'].values)

    # Generate all figures
    print(f'\nGenerating {args.format.upper()} figures...')
    fig_trajectory_3d(traj, cfg, ref, args.output_dir, args.format)
    fig_trajectory_2d(traj, cfg, ref, args.output_dir, args.format)
    fig_tracking_error(traj, ref, args.output_dir, args.format)
    fig_cable_tensions(tensions, args.output_dir, args.format)
    fig_estimation_error(traj, est, ref, args.output_dir, args.format)
    fig_safety_constraints(tensions, attitude_data, args.output_dir, args.format)
    fig_control_effort(ctrl, args.output_dir, args.format)
    fig_wind_disturbance(wind, args.output_dir, args.format)
    fig_attitude_tracking(attitude_data, args.output_dir, args.format)

    # Print summary statistics
    print_summary_stats(traj, ref, tensions, est, ctrl, wind, args.output_dir)

    print(f'\nAll figures saved to: {args.output_dir}/')
    print('Done.')


if __name__ == '__main__':
    main()
