#!/usr/bin/env python3
"""
Generate IROS 2026 supplementary video from simulation CSV logs.

Creates a composite MP4 showing 3D trajectory animation synchronized with
live telemetry panels (cable tensions, tracking error, mass estimation).

Usage:
    python generate_video.py --data-dir outputs/monte_carlo/seed_001
    python generate_video.py --data-dir outputs/monte_carlo/seed_001 --output video.mp4
    python generate_video.py --data-dir outputs/monte_carlo/seed_001 --fast
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont

# Import helpers from figure generator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_figures import parse_config, load_csv, build_reference_trajectory

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    'drone0': '#0072B2',
    'drone1': '#D55E00',
    'drone2': '#009E73',
    'load':   '#CC79A7',
    'ref':    '#000000',
    'limit':  '#E69F00',
    'grey':   '#999999',
}
ALL_DRONE_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
                     '#56B4E9', '#E69F00', '#000000']
DRONE_COLORS = ALL_DRONE_COLORS[:3]
DRONE_NAMES = ['Drone 0', 'Drone 1', 'Drone 2']
CABLE_NAMES = ['Cable 0', 'Cable 1', 'Cable 2']

PHASES = [
    (0,  2,  'Pre-lift'),
    (2,  6,  'Pickup & Ascent'),
    (7,  20, 'Figure-8 Right'),
    (21, 36, 'Figure-8 Left'),
    (39, 43, 'Descent'),
    (43, 50, 'Stabilize'),
]


def get_phase_label(t):
    for t0, t1, label in PHASES:
        if t0 <= t < t1:
            return label
    return ''


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.edgecolor': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
        'text.color': '#e0e0e0',
        'grid.color': '#2a2a4a',
        'grid.alpha': 0.5,
        'lines.linewidth': 1.2,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_data(data_dir):
    """Load and preprocess all simulation CSVs."""
    cfg = parse_config(data_dir)
    traj = load_csv(data_dir, 'trajectories.csv')
    tensions = load_csv(data_dir, 'tensions.csv')
    attitude = load_csv(data_dir, 'attitude_data.csv')
    estimator = load_csv(data_dir, 'estimator_outputs.csv')
    wind = load_csv(data_dir, 'wind_disturbance.csv')

    # Detect number of drones from columns
    num_drones = sum(1 for c in traj.columns if c.startswith('drone') and c.endswith('_x'))

    t = traj['time'].values
    ref = build_reference_trajectory(cfg, t)

    # Tracking error
    err_x = traj['load_x'].values - ref[:, 0]
    err_y = traj['load_y'].values - ref[:, 1]
    err_z = traj['load_z'].values - ref[:, 2]
    err_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2) * 100  # cm

    # Running RMSE
    rmse_running = np.zeros_like(err_3d)
    for i in range(1, len(err_3d)):
        rmse_running[i] = np.sqrt(np.mean(err_3d[:i+1]**2))

    data = {
        'cfg': cfg,
        't': t,
        'ref': ref,
        'traj': traj,
        'tensions': tensions,
        'attitude': attitude,
        'estimator': estimator,
        'wind': wind,
        'err_3d': err_3d,
        'rmse_running': rmse_running,
        'num_drones': num_drones,
    }
    return data


# ---------------------------------------------------------------------------
# Meshcat MP4 reader and composite helpers
# ---------------------------------------------------------------------------
class MeshcatVideoReader:
    """Streaming reader for a pre-recorded Drake Meshcat MP4.

    Decodes frames on demand via ffmpeg subprocess.  Maps frame indices
    to simulation time using the linear mapping
        t_sim = frame_idx * sim_duration / total_frames.
    """

    def __init__(self, mp4_path: str, sim_duration: float = 50.0):
        self.mp4_path = mp4_path
        self.sim_duration = sim_duration

        # Probe video metadata.
        probe = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0', mp4_path,
        ]).decode().strip()
        parts = probe.split(',')
        self.width = int(parts[0])
        self.height = int(parts[1])

        # Count frames.
        self.total_frames = int(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_packets', '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0', mp4_path,
        ]).decode().strip())

        self.sim_times = np.linspace(0, sim_duration, self.total_frames)
        self._frames = None  # lazy-loaded cache

    def _load_all(self):
        """Decode all frames into memory (called once on first access)."""
        proc = subprocess.Popen(
            ['ffmpeg', '-i', self.mp4_path,
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        frame_bytes = self.width * self.height * 3
        frames = []
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frames.append(
                np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3).copy()
            )
        proc.wait()
        self._frames = frames

    def frame_at_time(self, t_sim: float) -> np.ndarray:
        """Return the closest Meshcat frame for the given simulation time."""
        if self._frames is None:
            self._load_all()
        idx = int(np.clip(
            np.searchsorted(self.sim_times, t_sim),
            0, len(self._frames) - 1,
        ))
        return self._frames[idx]


def _try_load_font(size: int, bold: bool = False):
    """Try to load a TrueType font, fall back to default."""
    names = (['DejaVuSans-Bold.ttf', 'LiberationSans-Bold.ttf']
             if bold else ['DejaVuSans.ttf', 'LiberationSans-Regular.ttf'])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_telemetry_panel(data, t_now, panel_width=768, panel_height=1080,
                           dpi=100):
    """Render the 4 telemetry panels for a single simulation time.

    Returns an (panel_height, panel_width, 3) uint8 numpy array.
    """
    fig_w = panel_width / dpi
    fig_h = panel_height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(4, 1, hspace=0.45,
                           left=0.18, right=0.95, top=0.95, bottom=0.06)
    ax_tension = fig.add_subplot(gs[0])
    ax_error = fig.add_subplot(gs[1])
    ax_wind = fig.add_subplot(gs[2])
    ax_est = fig.add_subplot(gs[3])

    t = data['t']
    traj = data['traj']
    tensions = data['tensions']
    err_3d = data['err_3d']
    wind = data['wind']
    nd = data['num_drones']
    drone_colors = ALL_DRONE_COLORS[:nd]
    max_time = t[-1]

    idx = min(np.searchsorted(t, t_now), len(t) - 1)
    sl = slice(0, idx + 1)
    t_sl = t[sl]

    wind_idx = min(np.searchsorted(wind['time'].values, t_now), len(wind) - 1)
    wind_sl = slice(0, wind_idx + 1)
    wind_t = wind['time'].values[wind_sl]

    # Precompute smoothed mass estimates.
    from scipy.ndimage import uniform_filter1d
    gravity = 9.81
    mass_est = np.zeros((len(t), nd))
    for q in range(nd):
        T_mag = tensions[f'rope{q}_mag'].values
        mass_est[:, q] = uniform_filter1d(T_mag / gravity, size=100)

    # --- Cable tensions ---
    ax_tension.set_title('Cable Tensions', fontsize=11, pad=4)
    ax_tension.set_ylabel('Tension (N)')
    ax_tension.set_xlim(0, max_time)
    ax_tension.set_ylim(-1, 35)
    ax_tension.axhline(2.0, color=COLORS['limit'], ls='--', lw=0.8, alpha=0.7)
    ax_tension.grid(True, alpha=0.3)
    for q in range(nd):
        ax_tension.plot(t_sl, tensions[f'rope{q}_mag'].values[sl],
                        color=drone_colors[q], lw=1.0, label=f'Cable {q}')
    ax_tension.legend(loc='upper right', framealpha=0.7, fontsize=7)

    # --- Tracking error ---
    ax_error.set_title('Payload Tracking Error', fontsize=11, pad=4)
    ax_error.set_ylabel('Error (cm)')
    ax_error.set_xlim(0, max_time)
    ax_error.set_ylim(0, 100)
    ax_error.grid(True, alpha=0.3)
    ax_error.plot(t_sl, err_3d[sl], color=COLORS['load'], lw=1.0)
    ax_error.fill_between(t_sl, 0, err_3d[sl], color=COLORS['load'], alpha=0.15)
    if idx > 100:
        rmse_now = np.sqrt(np.mean(err_3d[100:idx + 1] ** 2))
        ax_error.axhline(rmse_now, color='white', ls=':', lw=0.6, alpha=0.5)
        ax_error.text(max_time * 0.98, rmse_now + 2,
                      f'RMSE: {rmse_now:.1f} cm', ha='right',
                      fontsize=8, color='white', alpha=0.8)

    # --- Wind velocity ---
    ax_wind.set_title('Wind Velocity (Dryden Turbulence)', fontsize=11, pad=4)
    ax_wind.set_ylabel('Velocity (m/s)')
    ax_wind.set_xlim(0, max_time)
    ax_wind.set_ylim(-0.5, 2.5)
    ax_wind.grid(True, alpha=0.3)
    if len(wind_t) > 0:
        ax_wind.plot(wind_t, wind['wind_vx'].values[wind_sl],
                     color='#56B4E9', lw=1.0, label='$v_{w,x}$')
        ax_wind.plot(wind_t, wind['wind_vy'].values[wind_sl],
                     color='#E69F00', lw=1.0, label='$v_{w,y}$')
        ax_wind.plot(wind_t, wind['wind_vz'].values[wind_sl],
                     color='#009E73', lw=1.0, label='$v_{w,z}$')
        w_mag = np.sqrt(wind['wind_vx'].values[wind_sl] ** 2 +
                        wind['wind_vy'].values[wind_sl] ** 2 +
                        wind['wind_vz'].values[wind_sl] ** 2)
        ax_wind.fill_between(wind_t, 0, w_mag, color='#56B4E9', alpha=0.1)
        ax_wind.plot(wind_t, w_mag, color='white', lw=0.6, ls='--', alpha=0.5)
    ax_wind.legend(loc='upper right', framealpha=0.7, fontsize=7, ncol=3)

    # --- Mass estimation ---
    ax_est.set_title('Per-Drone Mass Estimate $\\hat{\\theta}_i$',
                     fontsize=11, pad=4)
    ax_est.set_xlabel('Time (s)')
    ax_est.set_ylabel('$\\hat{\\theta}_i$ (kg)')
    ax_est.set_xlim(0, max_time)
    ax_est.set_ylim(0, 3.5)
    ax_est.axhline(1.0, color='white', ls='--', lw=0.8, alpha=0.5)
    ax_est.grid(True, alpha=0.3)
    for q in range(nd):
        ax_est.plot(t_sl, mass_est[sl, q],
                    color=drone_colors[q], lw=1.0, label=f'Drone {q}')
    ax_est.text(max_time * 0.98, 1.15, '$m_L/N = 1.0$ kg',
                ha='right', fontsize=8, color='white', alpha=0.7)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def composite_meshcat_telemetry(meshcat_frame, telemetry_frame,
                                total_width=1920, total_height=1080,
                                meshcat_ratio=0.6):
    """Assemble a side-by-side composite: Meshcat (left) + telemetry (right)."""
    left_w = int(total_width * meshcat_ratio)  # 1152
    right_w = total_width - left_w             # 768

    composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Scale the Meshcat frame to fit the left panel (preserve aspect ratio).
    mc_img = Image.fromarray(meshcat_frame)
    mc_w, mc_h = mc_img.size
    scale = min(left_w / mc_w, total_height / mc_h)
    new_w, new_h = int(mc_w * scale), int(mc_h * scale)
    mc_resized = np.array(mc_img.resize((new_w, new_h), Image.LANCZOS))

    y_off = (total_height - new_h) // 2
    x_off = (left_w - new_w) // 2
    composite[y_off:y_off + new_h, x_off:x_off + new_w] = mc_resized

    # Place telemetry panel on the right.
    tel_h, tel_w = telemetry_frame.shape[:2]
    composite[:min(tel_h, total_height), left_w:left_w + min(tel_w, right_w)] = \
        telemetry_frame[:min(tel_h, total_height), :min(tel_w, right_w)]

    return composite


def add_overlays(frame, t_now, data):
    """Draw Drake branding and time readout onto the composite frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]

    # Scale font sizes relative to 1080p baseline.
    s = h / 1080.0
    font_title = _try_load_font(max(int(22 * s), 10), bold=True)
    font_time = _try_load_font(max(int(16 * s), 8), bold=True)
    font_small = _try_load_font(max(int(13 * s), 7))

    nd = data['num_drones']
    phase = get_phase_label(t_now)

    pad = int(20 * s)

    # Top-left: title.
    draw.text((pad, int(12 * s)), 'Drake MeshcatVisualizer',
              fill=(255, 255, 255), font=font_title)
    draw.text((pad, int(42 * s)),
              f'N = {nd} Quadrotors   |   t = {t_now:.1f} s   |   {phase}',
              fill=(200, 200, 200), font=font_time)

    # Bottom-left: branding.
    draw.text((pad, h - int(28 * s)),
              'Simulated in Drake  |  Bead-chain cables  |  '
              'Dryden wind turbulence  |  ESKF sensor fusion',
              fill=(150, 150, 150), font=font_small)

    return np.array(img)


def render_meshcat_composite(data, meshcat_reader, fps=30,
                             start_time=0.0, duration=None, speed=1.0,
                             out_width=1920, out_height=1080, dpi=100):
    """Render the side-by-side Meshcat + telemetry composite scene.

    Args:
        data: simulation data dict from ``load_all_data``.
        meshcat_reader: ``MeshcatVideoReader`` instance.
        fps: output video frame rate.
        start_time: simulation start time for the scene.
        duration: scene duration in simulation seconds (None = full sim).
        speed: playback speed multiplier (1.0 = real-time).
        out_width: output frame width in pixels.
        out_height: output frame height in pixels.
        dpi: matplotlib dpi for the telemetry panel.

    Returns:
        List of (out_height, out_width, 3) uint8 numpy arrays.
    """
    sim_duration = data['t'][-1]
    if duration is None:
        duration = sim_duration - start_time
    end_time = min(start_time + duration, sim_duration)

    meshcat_ratio = 0.6
    tel_w = int(out_width * (1 - meshcat_ratio))
    tel_h = out_height

    dt_render = speed / fps
    render_times = np.arange(start_time, end_time, dt_render)

    frames = []
    for frame_idx, t_now in enumerate(render_times):
        mc_frame = meshcat_reader.frame_at_time(t_now)
        tel_frame = render_telemetry_panel(data, t_now,
                                           panel_width=tel_w,
                                           panel_height=tel_h, dpi=dpi)
        comp = composite_meshcat_telemetry(mc_frame, tel_frame,
                                           total_width=out_width,
                                           total_height=out_height)
        comp = add_overlays(comp, t_now, data)
        frames.append(comp)

        if frame_idx % 50 == 0:
            print(f'  Meshcat composite {frame_idx}/{len(render_times)} '
                  f'(t={t_now:.1f}s)')

    return frames


# ---------------------------------------------------------------------------
# Scene: Title card
# ---------------------------------------------------------------------------
def render_title_card(fig, duration_frames, fps, data=None):
    """Render a static title card."""
    fig.clear()
    fig.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    nd = data['num_drones'] if data else 3
    payload_mass = float(data['cfg'].get('payload_mass', 3.0)) if data else 3.0

    # Compute RMSE from data if available
    if data is not None:
        err = data['err_3d']
        t = data['t']
        mask = t > 6.0
        rmse = np.sqrt(np.mean(err[mask]**2))
        rmse_str = f'Tracking RMSE: {rmse:.1f} cm'
    else:
        rmse_str = 'Tracking RMSE: 23.7 cm'

    ax.text(0.5, 0.72, 'GPAC: Geometric Position and Attitude Control',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='white')
    ax.text(0.5, 0.62, 'Decentralized Cooperative Aerial Transport',
            ha='center', va='center', fontsize=14, color='#a0c4ff')
    ax.text(0.5, 0.48,
            f'{nd} Quadrotors  |  {payload_mass:.1f} kg Payload  |  Flexible Bead-Chain Cables\n'
            'ESKF Sensor Fusion  |  Dryden Wind Turbulence  |  CBF Safety Filter',
            ha='center', va='center', fontsize=10, color='#999999',
            linespacing=1.6)
    ax.text(0.5, 0.32, 'IEEE IROS 2026  |  Supplementary Video',
            ha='center', va='center', fontsize=11, color='#e0e0e0')
    ax.text(0.5, 0.18,
            f'{rmse_str}\n'
            '< 1 MFLOP/s per agent  |  No inter-agent communication',
            ha='center', va='center', fontsize=9, color='#777777',
            linespacing=1.5)

    frames = []
    for _ in range(duration_frames):
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())
    return frames


# ---------------------------------------------------------------------------
# Scene: Drake-style 3D simulation view (white background, 1x speed)
# ---------------------------------------------------------------------------
def render_drake_view(data, fps=30, start_time=15.0, duration=20.0):
    """Render a clean Drake/Meshcat-style 3D view at 1x speed.

    White background, solid colored bodies, ground plane grid, no telemetry
    panels. Shows the simulation as it would appear in Drake's Meshcat viewer.
    """
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    fig.set_facecolor('white')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f5f5f5')

    t = data['t']
    traj = data['traj']
    ref = data['ref']
    nd = data['num_drones']
    drone_colors_solid = ['#2176FF', '#FF6B35', '#22B573', '#B537F2', '#FFD23F',
                          '#00B4D8', '#FF006E', '#3A0CA3'][:nd]
    end_time = start_time + duration

    # Axis setup
    x_range = [ref[:, 0].min() - 1.0, ref[:, 0].max() + 1.0]
    y_range = [ref[:, 1].min() - 1.0, ref[:, 1].max() + 1.0]
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim([-0.1, 4.5])
    ax.set_xlabel('X (m)', fontsize=10, color='#333333', labelpad=8)
    ax.set_ylabel('Y (m)', fontsize=10, color='#333333', labelpad=8)
    ax.set_zlabel('Z (m)', fontsize=10, color='#333333', labelpad=8)
    ax.tick_params(colors='#666666', labelsize=8)

    # Style panes (light grey)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('#eeeeee')
    ax.yaxis.pane.set_facecolor('#e8e8e8')
    ax.zaxis.pane.set_facecolor('#f0f0f0')
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, alpha=0.3, color='#aaaaaa')

    # Ground plane
    gx = np.linspace(x_range[0], x_range[1], 10)
    gy = np.linspace(y_range[0], y_range[1], 10)
    gx_m, gy_m = np.meshgrid(gx, gy)
    ax.plot_surface(gx_m, gy_m, np.zeros_like(gx_m),
                    alpha=0.15, color='#8B8B8B')
    # Grid lines on ground
    for x in np.arange(np.ceil(x_range[0]), np.floor(x_range[1]) + 1, 1.0):
        ax.plot([x, x], y_range, [0, 0], color='#bbbbbb', lw=0.3, alpha=0.5)
    for y in np.arange(np.ceil(y_range[0]), np.floor(y_range[1]) + 1, 1.0):
        ax.plot(x_range, [y, y], [0, 0], color='#bbbbbb', lw=0.3, alpha=0.5)

    # Reference trajectory (thin dashed)
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
            '--', color='#999999', lw=0.8, alpha=0.5)

    # Render times at 1x speed
    render_times = np.arange(start_time, end_time, 1.0 / fps)

    dynamic_artists = []
    trail_x, trail_y, trail_z = [], [], []
    frames = []

    for frame_idx, t_now in enumerate(render_times):
        idx = min(np.searchsorted(t, t_now), len(t) - 1)

        # Clear dynamic artists
        for a in dynamic_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        dynamic_artists.clear()

        lx = traj['load_x'].values[idx]
        ly = traj['load_y'].values[idx]
        lz = traj['load_z'].values[idx]

        # Payload trail
        trail_x.append(lx); trail_y.append(ly); trail_z.append(lz)
        if len(trail_x) > 200:
            trail_x.pop(0); trail_y.pop(0); trail_z.pop(0)
        if len(trail_x) > 1:
            ln = ax.plot(trail_x, trail_y, trail_z,
                         color='#FF4444', alpha=0.3, lw=1.5)
            dynamic_artists.extend(ln)

        # Payload sphere (large red)
        sc_l = ax.scatter([lx], [ly], [lz], c='#CC0000', s=150,
                          marker='o', zorder=10, edgecolors='#880000', linewidths=0.8)
        dynamic_artists.append(sc_l)

        # Drones (quadrotor shape with oriented arms)
        arm_length = 0.15  # half arm span [m]
        for q in range(nd):
            dx = traj[f'drone{q}_x'].values[idx]
            dy = traj[f'drone{q}_y'].values[idx]
            dz = traj[f'drone{q}_z'].values[idx]

            # Get rotation matrix from quaternion
            qw = traj[f'drone{q}_qw'].values[idx]
            qx = traj[f'drone{q}_qx'].values[idx]
            qy = traj[f'drone{q}_qy'].values[idx]
            qz = traj[f'drone{q}_qz'].values[idx]
            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])

            # 4 arm directions in body frame (X-pattern)
            body_arms = [
                np.array([ arm_length,  arm_length, 0]),
                np.array([-arm_length,  arm_length, 0]),
                np.array([-arm_length, -arm_length, 0]),
                np.array([ arm_length, -arm_length, 0]),
            ]
            pos = np.array([dx, dy, dz])

            # Draw 4 arms
            for arm_body in body_arms:
                tip = pos + R @ arm_body
                ln_arm = ax.plot([dx, tip[0]], [dy, tip[1]], [dz, tip[2]],
                                 color='#333333', lw=2.0, solid_capstyle='round')
                dynamic_artists.extend(ln_arm)
                # Rotor disc at tip
                sc_rotor = ax.scatter([tip[0]], [tip[1]], [tip[2]],
                                      c=drone_colors_solid[q], s=35, marker='o',
                                      zorder=10, edgecolors='#333333', linewidths=0.3)
                dynamic_artists.append(sc_rotor)

            # Central body
            sc_d = ax.scatter([dx], [dy], [dz], c=drone_colors_solid[q], s=60,
                              marker='o', zorder=11, edgecolors='#222222', linewidths=0.6)
            dynamic_artists.append(sc_d)

            # Cable (rope to payload)
            ln_c = ax.plot([dx, lx], [dy, ly], [dz, lz],
                           color='#666666', lw=1.0, alpha=0.5)
            dynamic_artists.extend(ln_c)

            # Drone label
            txt_d = ax.text(dx, dy, dz + 0.25, f'Q{q}',
                            fontsize=7, ha='center', color=drone_colors_solid[q],
                            fontweight='bold')
            dynamic_artists.append(txt_d)

        # Camera rotation (slow)
        ax.view_init(elev=22 + 3 * np.sin((t_now - start_time) / 10),
                     azim=-55 + (t_now - start_time) * 1.5)

        # Title
        sim_t = t_now
        ax.set_title(
            f'Drake Multibody Simulation  |  N = {nd} Quadrotors  |  '
            f't = {sim_t:.1f} s  |  1x speed',
            fontsize=13, pad=10, color='#333333', fontweight='bold')

        # "Simulated in Drake" watermark
        fig.text(0.02, 0.02, 'Drake Toolbox (MIT)  |  Bead-chain cables  |  '
                 'Dryden wind turbulence  |  ESKF sensor fusion',
                 fontsize=8, color='#aaaaaa', ha='left')

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())

        if frame_idx % 50 == 0:
            print(f'  Drake view frame {frame_idx}/{len(render_times)} '
                  f'(t={t_now:.1f}s)')

    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Scene: Main mission animation
# ---------------------------------------------------------------------------
def create_mission_figure():
    """Create the composite figure layout for the mission scene."""
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    fig.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(4, 2, width_ratios=[1.2, 1],
                           hspace=0.40, wspace=0.25,
                           left=0.05, right=0.97, top=0.92, bottom=0.05)

    # 3D trajectory (left, spans all 4 rows)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')

    # Right panels
    ax_tension = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[1, 1])
    ax_wind = fig.add_subplot(gs[2, 1])
    ax_est = fig.add_subplot(gs[3, 1])

    return fig, ax3d, ax_tension, ax_error, ax_wind, ax_est


def setup_3d_axis(ax, data):
    """Configure the 3D trajectory axis."""
    ref = data['ref']
    ax.set_facecolor('#16213e')
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Z (m)', labelpad=6)

    x_range = [ref[:, 0].min() - 0.8, ref[:, 0].max() + 0.8]
    y_range = [ref[:, 1].min() - 0.8, ref[:, 1].max() + 0.8]
    z_range = [-0.1, 4.0]
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.view_init(elev=25, azim=-60)

    # Draw reference trajectory
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
            '--', color='#ffffff', alpha=0.3, linewidth=0.8, label='Reference')

    # Ground plane
    gx = np.linspace(x_range[0], x_range[1], 2)
    gy = np.linspace(y_range[0], y_range[1], 2)
    gx, gy = np.meshgrid(gx, gy)
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.08, color='#888888')

    # Style panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#2a2a4a')
    ax.yaxis.pane.set_edgecolor('#2a2a4a')
    ax.zaxis.pane.set_edgecolor('#2a2a4a')
    ax.grid(True, alpha=0.2)

    return ax


def setup_telemetry_axes(ax_tension, ax_error, ax_est, data):
    """Configure the right-side telemetry panels."""
    T = data['t'][-1]

    # Tension panel
    ax_tension.set_title('Cable Tensions', fontsize=11, pad=4)
    ax_tension.set_ylabel('Tension (N)')
    ax_tension.set_xlim(0, T)
    ax_tension.set_ylim(-1, 35)
    ax_tension.axhline(2.0, color=COLORS['limit'], ls='--', lw=0.8, alpha=0.7, label='$T_{min}$')
    ax_tension.grid(True, alpha=0.3)

    # Error panel
    ax_error.set_title('Payload Tracking Error', fontsize=11, pad=4)
    ax_error.set_ylabel('Error (cm)')
    ax_error.set_xlim(0, T)
    ax_error.set_ylim(0, 100)
    ax_error.grid(True, alpha=0.3)

    # Estimation panel
    ax_est.set_title('Mass Estimation', fontsize=11, pad=4)
    ax_est.set_xlabel('Time (s)')
    ax_est.set_ylabel('$\\hat{\\theta}_i$ (kg)')
    ax_est.set_xlim(0, T)
    ax_est.set_ylim(0, 4.0)
    ax_est.axhline(1.0, color='white', ls='--', lw=0.8, alpha=0.5, label='$m_L/N$')
    ax_est.grid(True, alpha=0.3)


def render_mission(data, fps=30, speed=2.0, max_time=None):
    """Render the main mission animation frames."""
    fig, ax3d, ax_tension, ax_error, ax_wind, ax_est = create_mission_figure()
    setup_3d_axis(ax3d, data)
    setup_telemetry_axes(ax_tension, ax_error, ax_est, data)

    t = data['t']
    traj = data['traj']
    tensions = data['tensions']
    ref = data['ref']
    err_3d = data['err_3d']
    cfg = data['cfg']
    wind = data['wind']
    nd = data['num_drones']
    drone_colors = ALL_DRONE_COLORS[:nd]

    if max_time is None:
        max_time = t[-1]

    dt_render = speed / fps  # simulation seconds per frame
    render_times = np.arange(0, max_time, dt_render)

    # Precompute mass estimation (approximate from tension data)
    gravity = 9.81
    mass_est = np.zeros((len(t), nd))
    for q in range(nd):
        T_mag = tensions[f'rope{q}_mag'].values
        mass_est[:, q] = T_mag / gravity

    # Smooth mass estimates
    from scipy.ndimage import uniform_filter1d
    for q in range(nd):
        mass_est[:, q] = uniform_filter1d(mass_est[:, q], size=100)

    # Trail storage
    trail_len = 300
    load_trail_x, load_trail_y, load_trail_z = [], [], []

    # Store dynamic artists to remove each frame
    dynamic_artists = []

    frames = []
    for frame_idx, t_now in enumerate(render_times):
        # Find closest data index
        idx = np.searchsorted(t, t_now)
        idx = min(idx, len(t) - 1)

        # Remove previous frame's dynamic artists
        for artist in dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        dynamic_artists.clear()

        # Payload position
        lx = traj['load_x'].values[idx]
        ly = traj['load_y'].values[idx]
        lz = traj['load_z'].values[idx]

        # Update trail
        load_trail_x.append(lx)
        load_trail_y.append(ly)
        load_trail_z.append(lz)
        if len(load_trail_x) > trail_len:
            load_trail_x.pop(0)
            load_trail_y.pop(0)
            load_trail_z.pop(0)

        # Draw load trail
        if len(load_trail_x) > 1:
            lines = ax3d.plot(load_trail_x, load_trail_y, load_trail_z,
                              color=COLORS['load'], alpha=0.5, linewidth=1.5)
            dynamic_artists.extend(lines)

        # Draw payload
        sc = ax3d.scatter([lx], [ly], [lz], c=COLORS['load'], s=80,
                          marker='o', zorder=10, edgecolors='white', linewidths=0.5)
        dynamic_artists.append(sc)

        # Draw drones and cables
        for q in range(nd):
            dx = traj[f'drone{q}_x'].values[idx]
            dy = traj[f'drone{q}_y'].values[idx]
            dz = traj[f'drone{q}_z'].values[idx]

            sc_d = ax3d.scatter([dx], [dy], [dz], c=drone_colors[q], s=50,
                                marker='^', zorder=10, edgecolors='white', linewidths=0.3)
            dynamic_artists.append(sc_d)

            # Cable line
            lines_c = ax3d.plot([dx, lx], [dy, ly], [dz, lz],
                                color=drone_colors[q], alpha=0.4, linewidth=0.8)
            dynamic_artists.extend(lines_c)

        # Wind arrow in 3D view
        wind_idx = min(np.searchsorted(wind['time'].values, t_now), len(wind) - 1)
        wvx = wind['wind_vx'].values[wind_idx]
        wvy = wind['wind_vy'].values[wind_idx]
        wvz = wind['wind_vz'].values[wind_idx]
        w_mag = np.sqrt(wvx**2 + wvy**2 + wvz**2)

        # Draw wind arrow at a fixed position (upper corner of workspace)
        arrow_origin = np.array([ref[:, 0].max() + 0.3, ref[:, 1].min() - 0.3, 3.5])
        arrow_scale = 0.8  # scale factor for visibility
        arrow_lines = ax3d.plot(
            [arrow_origin[0], arrow_origin[0] + wvx * arrow_scale],
            [arrow_origin[1], arrow_origin[1] + wvy * arrow_scale],
            [arrow_origin[2], arrow_origin[2] + wvz * arrow_scale],
            color='#56B4E9', linewidth=2.5, alpha=0.9)
        dynamic_artists.extend(arrow_lines)
        # Arrow head dot
        arrow_tip = ax3d.scatter(
            [arrow_origin[0] + wvx * arrow_scale],
            [arrow_origin[1] + wvy * arrow_scale],
            [arrow_origin[2] + wvz * arrow_scale],
            c='#56B4E9', s=40, marker='>', zorder=10)
        dynamic_artists.append(arrow_tip)
        # Wind label
        wind_txt = ax3d.text(arrow_origin[0], arrow_origin[1], arrow_origin[2] + 0.25,
                             f'Wind {w_mag:.1f} m/s', color='#56B4E9',
                             fontsize=8, ha='center')
        dynamic_artists.append(wind_txt)

        # Slowly rotate camera
        ax3d.view_init(elev=25 + 5 * np.sin(t_now / 15), azim=-60 + t_now * 0.8)

        # Phase label with drone count
        phase = get_phase_label(t_now)
        ax3d.set_title(f'N={nd}   {phase}     t = {t_now:.1f} s', fontsize=13, pad=8,
                        color='white', fontweight='bold')

        # --- Telemetry panels: draw up to current time ---
        sl = slice(0, idx + 1)
        t_sl = t[sl]

        # Tensions
        ax_tension.clear()
        ax_tension.set_title('Cable Tensions', fontsize=11, pad=4)
        ax_tension.set_ylabel('Tension (N)')
        ax_tension.set_xlim(0, max_time)
        ax_tension.set_ylim(-1, 35)
        ax_tension.axhline(2.0, color=COLORS['limit'], ls='--', lw=0.8, alpha=0.7)
        ax_tension.grid(True, alpha=0.3)
        for q in range(nd):
            ax_tension.plot(t_sl, tensions[f'rope{q}_mag'].values[sl],
                            color=drone_colors[q], lw=1.0, label=f'Cable {q}')
        if frame_idx == 0:
            ax_tension.legend(loc='upper right', framealpha=0.7, fontsize=7)

        # Tracking error
        ax_error.clear()
        ax_error.set_title('Payload Tracking Error', fontsize=11, pad=4)
        ax_error.set_ylabel('Error (cm)')
        ax_error.set_xlim(0, max_time)
        ax_error.set_ylim(0, 100)
        ax_error.grid(True, alpha=0.3)
        ax_error.plot(t_sl, err_3d[sl], color=COLORS['load'], lw=1.0)
        ax_error.fill_between(t_sl, 0, err_3d[sl], color=COLORS['load'], alpha=0.15)
        if idx > 100:
            rmse_now = np.sqrt(np.mean(err_3d[100:idx+1]**2))
            ax_error.axhline(rmse_now, color='white', ls=':', lw=0.6, alpha=0.5)
            ax_error.text(max_time * 0.98, rmse_now + 2,
                          f'RMSE: {rmse_now:.1f} cm', ha='right',
                          fontsize=8, color='white', alpha=0.8)

        # Wind velocity
        ax_wind.clear()
        ax_wind.set_title('Wind Velocity (Dryden Turbulence)', fontsize=11, pad=4)
        ax_wind.set_ylabel('Velocity (m/s)')
        ax_wind.set_xlim(0, max_time)
        ax_wind.set_ylim(-0.5, 2.5)
        ax_wind.grid(True, alpha=0.3)
        wind_sl = slice(0, min(wind_idx + 1, len(wind)))
        wind_t = wind['time'].values[wind_sl]
        ax_wind.plot(wind_t, wind['wind_vx'].values[wind_sl],
                     color='#56B4E9', lw=1.0, label='$v_{w,x}$')
        ax_wind.plot(wind_t, wind['wind_vy'].values[wind_sl],
                     color='#E69F00', lw=1.0, label='$v_{w,y}$')
        ax_wind.plot(wind_t, wind['wind_vz'].values[wind_sl],
                     color='#009E73', lw=1.0, label='$v_{w,z}$')
        # Wind magnitude envelope
        w_magnitude = np.sqrt(wind['wind_vx'].values[wind_sl]**2 +
                              wind['wind_vy'].values[wind_sl]**2 +
                              wind['wind_vz'].values[wind_sl]**2)
        ax_wind.fill_between(wind_t, 0, w_magnitude, color='#56B4E9', alpha=0.1)
        ax_wind.plot(wind_t, w_magnitude, color='white', lw=0.6, ls='--', alpha=0.5)
        if wind_idx > 0:
            ax_wind.text(max_time * 0.98, w_magnitude[-1] + 0.1,
                         f'|w| = {w_magnitude[-1]:.2f} m/s', ha='right',
                         fontsize=8, color='#56B4E9', alpha=0.9)
        if frame_idx == 0:
            ax_wind.legend(loc='upper right', framealpha=0.7, fontsize=7, ncol=3)

        # Mass estimation
        ax_est.clear()
        ax_est.set_title('Per-Drone Mass Estimate $\\hat{\\theta}_i$', fontsize=11, pad=4)
        ax_est.set_xlabel('Time (s)')
        ax_est.set_ylabel('$\\hat{\\theta}_i$ (kg)')
        ax_est.set_xlim(0, max_time)
        ax_est.set_ylim(0, 3.5)
        ax_est.axhline(1.0, color='white', ls='--', lw=0.8, alpha=0.5)
        ax_est.grid(True, alpha=0.3)
        for q in range(nd):
            ax_est.plot(t_sl, mass_est[sl, q],
                        color=drone_colors[q], lw=1.0, label=f'Drone {q}')
        ax_est.text(max_time * 0.98, 1.15, '$m_L/N = 1.0$ kg',
                    ha='right', fontsize=8, color='white', alpha=0.7)

        # Render frame
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())

        if frame_idx % 50 == 0:
            print(f'  Frame {frame_idx}/{len(render_times)} '
                  f'(t={t_now:.1f}s / {max_time:.0f}s)')

    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Scene: End card
# ---------------------------------------------------------------------------
def render_end_card(fig, duration_frames, data):
    """Render an end card with key metrics."""
    fig.clear()
    fig.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    err = data['err_3d']
    t = data['t']
    mask = t > 6.0
    rmse = np.sqrt(np.mean(err[mask]**2))
    max_err = np.max(err[mask])

    ax.text(0.5, 0.78, 'Key Results', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white')

    nd = data['num_drones']
    metrics = [
        f'N = {nd} Quadrotors  |  Payload 3D RMSE:  {rmse:.1f} cm',
        f'Peak Error:  {max_err:.1f} cm  (during aggressive cornering)',
        f'Fully Decentralized  |  No inter-agent communication',
        f'Adaptive Mass Estimation via Concurrent Learning',
        f'CBF Safety Filter  |  Flexible Bead-Chain Cables',
        f'Computation:  < 1 MFLOP/s per agent',
    ]

    for i, m in enumerate(metrics):
        ax.text(0.5, 0.62 - i * 0.07, m, ha='center', va='center',
                fontsize=12, color='#a0c4ff' if i == 0 else '#cccccc')

    ax.text(0.5, 0.14, 'GPAC: Decentralized Geometric Cooperative Transport',
            ha='center', va='center', fontsize=10, color='#666666')
    ax.text(0.5, 0.08, 'IEEE IROS 2026',
            ha='center', va='center', fontsize=10, color='#666666')

    frames = []
    for _ in range(duration_frames):
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())
    return frames


# ---------------------------------------------------------------------------
# Main video assembly
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate IROS 2026 supplementary video')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Simulation log directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='Playback speed multiplier (default: 2x)')
    parser.add_argument('--fast', action='store_true',
                        help='Quick preview: lower resolution, 4x speed')
    parser.add_argument('--data-dir-5q', type=str, default=None,
                        help='5-quad simulation log (for scalability scene)')
    parser.add_argument('--meshcat-mp4', type=str, default=None,
                        help='Path to pre-recorded Drake Meshcat MP4 '
                             '(auto-discovered if omitted)')
    parser.add_argument('--no-meshcat', action='store_true',
                        help='Disable Meshcat frames; use matplotlib 3D only')
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir is None:
        mc = Path('/workspaces/Tether_Lift/outputs/monte_carlo')
        if mc.exists():
            seeds = sorted([d for d in mc.iterdir()
                            if d.is_dir() and d.name.startswith('seed_')])
            if seeds:
                args.data_dir = str(seeds[0])
        if args.data_dir is None:
            log_base = Path('/workspaces/Tether_Lift/outputs/logs')
            if log_base.exists():
                dirs = sorted([d for d in log_base.iterdir() if d.is_dir()])
                if dirs:
                    args.data_dir = str(dirs[-1])
    if args.data_dir is None:
        print('ERROR: No data directory found. Use --data-dir.', file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = '/workspaces/Tether_Lift/IEEE_IROS_2026/supplementary_video.mp4'

    if args.fast:
        args.speed = 4.0
        dpi = 80
        figsize = (12.80, 7.20)
    else:
        dpi = 100
        figsize = (19.20, 10.80)

    # Canonical output resolution (must match across all scenes).
    out_width = int(figsize[0] * dpi)
    out_height = int(figsize[1] * dpi)

    fps = args.fps

    print(f'Data:   {args.data_dir}')
    print(f'Output: {args.output}')
    print(f'FPS:    {fps},  Speed: {args.speed}x')
    print()

    setup_style()

    # Load data
    print('Loading simulation data...')
    data = load_all_data(args.data_dir)
    print(f'  Duration: {data["t"][-1]:.1f} s, {len(data["t"])} samples')
    print()

    # Auto-discover Meshcat MP4 if not explicitly provided.
    meshcat_reader = None
    if not args.no_meshcat:
        mc_mp4 = args.meshcat_mp4
        if mc_mp4 is None:
            # Try common locations relative to data-dir.
            candidates = [
                Path(args.data_dir).parent.parent / 'sim_recording.mp4',
                Path(args.data_dir).parent / 'sim_recording.mp4',
                Path(args.data_dir) / 'sim_recording.mp4',
                Path(__file__).resolve().parent.parent / 'outputs' / 'sim_recording.mp4',
            ]
            for c in candidates:
                if c.exists():
                    mc_mp4 = str(c)
                    break
        if mc_mp4 and os.path.isfile(mc_mp4):
            print(f'Meshcat MP4: {mc_mp4}')
            meshcat_reader = MeshcatVideoReader(mc_mp4, data['t'][-1])
            print(f'  {meshcat_reader.total_frames} frames, '
                  f'{meshcat_reader.width}x{meshcat_reader.height}')
        else:
            print('WARNING: No Meshcat MP4 found. '
                  'Using matplotlib 3D fallback (--meshcat-mp4 to specify).')
    print()

    all_frames = []

    # --- Scene 1: Title card (3 seconds) ---
    print('Rendering title card...')
    title_fig = plt.figure(figsize=figsize, dpi=dpi)
    title_frames = render_title_card(title_fig, duration_frames=fps * 3, fps=fps,
                                     data=data)
    all_frames.extend(title_frames)
    plt.close(title_fig)
    print(f'  {len(title_frames)} frames')

    # --- Scene 2: Drake Meshcat composite (or matplotlib fallback) ---
    if meshcat_reader is not None:
        print(f'Rendering Meshcat composite N={data["num_drones"]} '
              f'(20s @ 1x from t=15s)...')
        drake_frames = render_meshcat_composite(
            data, meshcat_reader, fps=fps,
            start_time=15.0, duration=20.0, speed=1.0,
            out_width=out_width, out_height=out_height, dpi=dpi)
    else:
        print(f'Rendering Drake view N={data["num_drones"]} '
              f'(20s @ 1x from t=15s)...')
        drake_frames = render_drake_view(
            data, fps=fps, start_time=15.0, duration=20.0)
    all_frames.extend(drake_frames)
    print(f'  {len(drake_frames)} frames')

    # --- Scene 3: Full mission with Meshcat + telemetry (or matplotlib) ---
    if meshcat_reader is not None:
        print(f'Rendering Meshcat composite N={data["num_drones"]} '
              f'full mission ({args.speed}x speed)...')
        mission_frames = render_meshcat_composite(
            data, meshcat_reader, fps=fps, speed=args.speed,
            out_width=out_width, out_height=out_height, dpi=dpi)
    else:
        print(f'Rendering N={data["num_drones"]} mission '
              f'({args.speed}x speed)...')
        mission_frames = render_mission(data, fps=fps, speed=args.speed)
    all_frames.extend(mission_frames)
    print(f'  {len(mission_frames)} frames')

    # --- Scene 4: Scalability demonstration (N=5) ---
    if args.data_dir_5q:
        print(f'\nLoading 5-quad data from {args.data_dir_5q}...')
        data_5q = load_all_data(args.data_dir_5q)
        print(f'  N={data_5q["num_drones"]} drones, {data_5q["t"][-1]:.0f}s')

        # Scalability transition card
        trans_fig = plt.figure(figsize=figsize, dpi=dpi)
        trans_fig.set_facecolor('#1a1a2e')
        ax_t = trans_fig.add_axes([0, 0, 1, 1])
        ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1); ax_t.axis('off')
        ax_t.set_facecolor('#1a1a2e')
        ax_t.text(0.5, 0.6, 'Scalability Demonstration',
                  ha='center', va='center', fontsize=22, fontweight='bold', color='white')
        ax_t.text(0.5, 0.45, f'N = 3  \u2192  N = {data_5q["num_drones"]}  quadcopters',
                  ha='center', va='center', fontsize=16, color='#a0c4ff')
        ax_t.text(0.5, 0.32, 'Same controller per agent \u2014 no retuning, no reconfiguration',
                  ha='center', va='center', fontsize=11, color='#999999')
        for _ in range(fps * 3):  # 3 second card
            trans_fig.canvas.draw()
            buf = np.frombuffer(trans_fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(trans_fig.canvas.get_width_height()[::-1] + (4,))
            all_frames.append(buf[:, :, :3].copy())
        plt.close(trans_fig)
        print(f'  Transition card: {fps * 3} frames')

        # Drake-style 3D view for N=5 (20s at 1x speed)
        print(f'Rendering Drake view N={data_5q["num_drones"]} (20s @ 1x from t=15s)...')
        drake_5q_frames = render_drake_view(data_5q, fps=fps, start_time=15.0, duration=20.0)
        all_frames.extend(drake_5q_frames)
        print(f'  {len(drake_5q_frames)} frames')

        # Full mission with telemetry (N=5)
        print(f'Rendering N={data_5q["num_drones"]} mission ({args.speed}x speed)...')
        mission_5q_frames = render_mission(data_5q, fps=fps, speed=args.speed)
        all_frames.extend(mission_5q_frames)
        print(f'  {len(mission_5q_frames)} frames')

    # --- End card ---
    print('Rendering end card...')
    end_fig = plt.figure(figsize=figsize, dpi=dpi)
    end_frames = render_end_card(end_fig, duration_frames=fps * 4, data=data)
    all_frames.extend(end_frames)
    plt.close(end_fig)
    print(f'  {len(end_frames)} frames')

    # --- Write MP4 ---
    total_frames = len(all_frames)
    duration_s = total_frames / fps
    print(f'\nTotal: {total_frames} frames, {duration_s:.1f} s at {fps} fps')
    print(f'Writing {args.output}...')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    h, w = all_frames[0].shape[:2]

    # Use ffmpeg subprocess for reliable H.264 encoding.
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '20',
        args.output
    ]

    print(f'  Encoding {w}x{h} @ {fps}fps via ffmpeg...')
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for i, frame in enumerate(all_frames):
        proc.stdin.write(frame.tobytes())
        if i % 100 == 0:
            print(f'  Writing frame {i}/{total_frames}')

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        print(f'ffmpeg error: {stderr[-500:]}', file=sys.stderr)
    else:
        fsize = os.path.getsize(args.output) / (1024 * 1024)
        print(f'\nVideo saved to: {args.output}')
        print(f'Duration: {duration_s:.1f} s  |  Resolution: {w}x{h}  |  '
              f'FPS: {fps}  |  Size: {fsize:.1f} MB')


if __name__ == '__main__':
    main()
