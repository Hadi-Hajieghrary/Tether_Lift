#include "load_trajectory_generator.h"

#include <algorithm>
#include <cmath>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;

// === MinJerkCoeffs implementation ===

void LoadTrajectoryGenerator::MinJerkCoeffs::Evaluate(
    double t, double& pos, double& vel, double& accel) const {
  // Normalize time to [0, 1]
  double tau = std::clamp(t / duration, 0.0, 1.0);
  double tau2 = tau * tau;
  double tau3 = tau2 * tau;
  double tau4 = tau3 * tau;
  double tau5 = tau4 * tau;

  pos = a0 + a1 * tau + a2 * tau2 + a3 * tau3 + a4 * tau4 + a5 * tau5;

  // Velocity (derivative w.r.t. actual time)
  double vel_norm = a1 + 2 * a2 * tau + 3 * a3 * tau2 + 4 * a4 * tau3 + 5 * a5 * tau4;
  vel = vel_norm / duration;

  // Acceleration (second derivative w.r.t. actual time)
  double accel_norm = 2 * a2 + 6 * a3 * tau + 12 * a4 * tau2 + 20 * a5 * tau3;
  accel = accel_norm / (duration * duration);
}

LoadTrajectoryGenerator::MinJerkCoeffs LoadTrajectoryGenerator::ComputeMinJerkSegment(
    double p0, double v0, double a0,
    double pf, double vf, double af,
    double duration) const {

  MinJerkCoeffs c;
  c.duration = duration;

  // Scale velocities and accelerations to normalized time
  double T = duration;
  double T2 = T * T;
  double v0_n = v0 * T;
  double vf_n = vf * T;
  double a0_n = a0 * T2;
  double af_n = af * T2;

  // Coefficients for 5th-order polynomial with boundary conditions
  // p(0) = p0, p(1) = pf
  // v(0) = v0_n, v(1) = vf_n  (normalized)
  // a(0) = a0_n, a(1) = af_n  (normalized)
  c.a0 = p0;
  c.a1 = v0_n;
  c.a2 = 0.5 * a0_n;
  c.a3 = 10 * (pf - p0) - 6 * v0_n - 4 * vf_n - 1.5 * a0_n + 0.5 * af_n;
  c.a4 = -15 * (pf - p0) + 8 * v0_n + 7 * vf_n + 1.5 * a0_n - af_n;
  c.a5 = 6 * (pf - p0) - 3 * (v0_n + vf_n) - 0.5 * (a0_n - af_n);

  return c;
}

// === LoadTrajectoryGenerator implementation ===

LoadTrajectoryGenerator::LoadTrajectoryGenerator(
    const std::vector<LoadWaypoint>& waypoints,
    const LoadTrajectoryParams& params)
    : params_(params), waypoints_(waypoints), total_duration_(0.0) {
  Initialize();
}

LoadTrajectoryGenerator LoadTrajectoryGenerator::CreateAltitudeRamp(
    const Eigen::Vector3d& initial_pos,
    const Eigen::Vector3d& final_pos,
    double start_time,
    const LoadTrajectoryParams& params) {

  // Compute duration based on distance and velocity limit
  double distance = (final_pos - initial_pos).norm();
  double cruise_vel = params.max_velocity;

  // Time for trapezoidal profile: accel phase + cruise + decel phase
  double accel_time = cruise_vel / params.max_acceleration;
  double accel_dist = 0.5 * params.max_acceleration * accel_time * accel_time;

  double cruise_dist = std::max(0.0, distance - 2 * accel_dist);
  double cruise_time = cruise_dist / cruise_vel;

  double total_move_time = 2 * accel_time + cruise_time;

  std::vector<LoadWaypoint> waypoints;

  // Initial hold
  LoadWaypoint wp0;
  wp0.position = initial_pos;
  wp0.arrival_time = 0.0;
  wp0.hold_time = start_time;
  waypoints.push_back(wp0);

  // Final position
  LoadWaypoint wp1;
  wp1.position = final_pos;
  wp1.arrival_time = start_time + total_move_time;
  wp1.hold_time = 1e6;  // Hold indefinitely
  waypoints.push_back(wp1);

  return LoadTrajectoryGenerator(waypoints, params);
}

LoadTrajectoryGenerator LoadTrajectoryGenerator::CreateHover(
    const Eigen::Vector3d& hover_position) {

  std::vector<LoadWaypoint> waypoints;
  LoadWaypoint wp;
  wp.position = hover_position;
  wp.arrival_time = 0.0;
  wp.hold_time = 1e9;  // Effectively infinite
  waypoints.push_back(wp);

  return LoadTrajectoryGenerator(waypoints, LoadTrajectoryParams());
}

void LoadTrajectoryGenerator::Initialize() {
  // Build trajectory segments
  BuildTrajectorySegments();

  // Declare output ports
  position_port_ =
      DeclareVectorOutputPort("load_position_des", BasicVector<double>(3),
                              &LoadTrajectoryGenerator::CalcPosition)
          .get_index();

  velocity_port_ =
      DeclareVectorOutputPort("load_velocity_des", BasicVector<double>(3),
                              &LoadTrajectoryGenerator::CalcVelocity)
          .get_index();

  acceleration_port_ =
      DeclareVectorOutputPort("load_acceleration_des", BasicVector<double>(3),
                              &LoadTrajectoryGenerator::CalcAcceleration)
          .get_index();

  complete_port_ =
      DeclareVectorOutputPort("trajectory_complete", BasicVector<double>(1),
                              &LoadTrajectoryGenerator::CalcComplete)
          .get_index();
}

void LoadTrajectoryGenerator::BuildTrajectorySegments() {
  segments_.clear();

  if (waypoints_.empty()) {
    // No waypoints - create a single stationary segment at origin
    TrajectorySegment seg;
    seg.start_time = 0.0;
    seg.end_time = 1e9;
    seg.type = SegmentType::kHold;
    seg.start_pos = Eigen::Vector3d::Zero();
    seg.end_pos = Eigen::Vector3d::Zero();
    seg.start_vel = Eigen::Vector3d::Zero();
    seg.end_vel = Eigen::Vector3d::Zero();
    segments_.push_back(seg);
    total_duration_ = 1e9;
    return;
  }

  double current_time = 0.0;

  for (size_t i = 0; i < waypoints_.size(); ++i) {
    const auto& wp = waypoints_[i];

    if (i == 0) {
      // First waypoint: hold segment if hold_time > 0
      if (wp.hold_time > 0) {
        TrajectorySegment hold_seg;
        hold_seg.start_time = 0.0;
        hold_seg.end_time = wp.hold_time;
        hold_seg.type = SegmentType::kHold;
        hold_seg.start_pos = wp.position;
        hold_seg.end_pos = wp.position;
        hold_seg.start_vel = Eigen::Vector3d::Zero();
        hold_seg.end_vel = Eigen::Vector3d::Zero();
        segments_.push_back(hold_seg);
        current_time = wp.hold_time;
      }
    } else {
      // Transition from previous waypoint to this one
      const auto& prev_wp = waypoints_[i - 1];

      // Motion segment
      TrajectorySegment move_seg;
      move_seg.start_time = current_time;
      move_seg.end_time = wp.arrival_time;
      move_seg.type = SegmentType::kMinJerk;
      move_seg.start_pos = prev_wp.position;
      move_seg.end_pos = wp.position;
      move_seg.start_vel = Eigen::Vector3d::Zero();  // Start from rest
      move_seg.end_vel = Eigen::Vector3d::Zero();    // End at rest

      // Compute minimum-jerk coefficients for each axis
      double seg_duration = move_seg.end_time - move_seg.start_time;
      if (seg_duration > 1e-6) {
        for (int axis = 0; axis < 3; ++axis) {
          move_seg.min_jerk_coeffs[axis] = ComputeMinJerkSegment(
              move_seg.start_pos[axis], 0.0, 0.0,
              move_seg.end_pos[axis], 0.0, 0.0,
              seg_duration);
        }
        segments_.push_back(move_seg);
      }
      current_time = wp.arrival_time;

      // Hold segment at this waypoint
      if (wp.hold_time > 0) {
        TrajectorySegment hold_seg;
        hold_seg.start_time = current_time;
        hold_seg.end_time = current_time + wp.hold_time;
        hold_seg.type = SegmentType::kHold;
        hold_seg.start_pos = wp.position;
        hold_seg.end_pos = wp.position;
        hold_seg.start_vel = Eigen::Vector3d::Zero();
        hold_seg.end_vel = Eigen::Vector3d::Zero();
        segments_.push_back(hold_seg);
        current_time += wp.hold_time;
      }
    }
  }

  total_duration_ = current_time;

  // Ensure we have at least one segment
  if (segments_.empty()) {
    TrajectorySegment seg;
    seg.start_time = 0.0;
    seg.end_time = 1e9;
    seg.type = SegmentType::kHold;
    seg.start_pos = waypoints_.empty() ? Eigen::Vector3d::Zero() : waypoints_[0].position;
    seg.end_pos = seg.start_pos;
    seg.start_vel = Eigen::Vector3d::Zero();
    seg.end_vel = Eigen::Vector3d::Zero();
    segments_.push_back(seg);
  }
}

int LoadTrajectoryGenerator::FindSegmentIndex(double t) const {
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (t >= segments_[i].start_time && t < segments_[i].end_time) {
      return static_cast<int>(i);
    }
  }
  // After all segments - return last
  return static_cast<int>(segments_.size()) - 1;
}

void LoadTrajectoryGenerator::EvaluateTrajectory(
    double t,
    Eigen::Vector3d& position,
    Eigen::Vector3d& velocity,
    Eigen::Vector3d& acceleration) const {

  int seg_idx = FindSegmentIndex(t);
  const auto& seg = segments_[seg_idx];

  switch (seg.type) {
    case SegmentType::kHold:
      position = seg.start_pos;
      velocity = Eigen::Vector3d::Zero();
      acceleration = Eigen::Vector3d::Zero();
      break;

    case SegmentType::kMinJerk: {
      double local_t = t - seg.start_time;
      for (int axis = 0; axis < 3; ++axis) {
        double p, v, a;
        seg.min_jerk_coeffs[axis].Evaluate(local_t, p, v, a);
        position[axis] = p;
        velocity[axis] = v;
        acceleration[axis] = a;
      }
      break;
    }

    case SegmentType::kLinear: {
      double local_t = t - seg.start_time;
      double duration = seg.end_time - seg.start_time;
      double alpha = (duration > 1e-6) ? (local_t / duration) : 1.0;
      alpha = std::clamp(alpha, 0.0, 1.0);

      position = seg.start_pos + alpha * (seg.end_pos - seg.start_pos);
      velocity = (seg.end_pos - seg.start_pos) / std::max(duration, 1e-6);
      acceleration = Eigen::Vector3d::Zero();
      break;
    }

    case SegmentType::kTrapezoidal:
      // TODO: Implement trapezoidal velocity profile
      position = seg.end_pos;
      velocity = Eigen::Vector3d::Zero();
      acceleration = Eigen::Vector3d::Zero();
      break;
  }
}

void LoadTrajectoryGenerator::CalcPosition(
    const Context<double>& context,
    BasicVector<double>* output) const {
  Eigen::Vector3d pos, vel, accel;
  EvaluateTrajectory(context.get_time(), pos, vel, accel);
  output->SetFromVector(pos);
}

void LoadTrajectoryGenerator::CalcVelocity(
    const Context<double>& context,
    BasicVector<double>* output) const {
  Eigen::Vector3d pos, vel, accel;
  EvaluateTrajectory(context.get_time(), pos, vel, accel);
  output->SetFromVector(vel);
}

void LoadTrajectoryGenerator::CalcAcceleration(
    const Context<double>& context,
    BasicVector<double>* output) const {
  Eigen::Vector3d pos, vel, accel;
  EvaluateTrajectory(context.get_time(), pos, vel, accel);
  output->SetFromVector(accel);
}

void LoadTrajectoryGenerator::CalcComplete(
    const Context<double>& context,
    BasicVector<double>* output) const {
  double t = context.get_time();
  double complete = (t >= total_duration_) ? 1.0 : 0.0;
  output->SetAtIndex(0, complete);
}

}  // namespace quad_rope_lift
