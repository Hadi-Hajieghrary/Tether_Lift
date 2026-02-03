#include "extended_load_trajectory_generator.h"
#include <cmath>
#include <algorithm>

namespace tether_lift {

ExtendedLoadTrajectoryGenerator::ExtendedLoadTrajectoryGenerator(
    const std::vector<quad_rope_lift::LoadWaypoint>& waypoints,
    const quad_rope_lift::LoadTrajectoryParams& traj_params,
    const FeasibilityParams& feas_params)
    : feas_params_(feas_params) {
  // Create base generator
  base_generator_ = std::make_unique<quad_rope_lift::LoadTrajectoryGenerator>(
      waypoints, traj_params);

  // Full trajectory output: [position(3), velocity(3), acceleration(3), jerk(3)]
  full_trajectory_port_ = DeclareVectorOutputPort(
      "full_trajectory", 12,
      &ExtendedLoadTrajectoryGenerator::CalcFullTrajectory).get_index();

  // Feasibility flag
  feasibility_port_ = DeclareVectorOutputPort(
      "feasibility", 1,
      &ExtendedLoadTrajectoryGenerator::CalcFeasibility).get_index();

  // Maximum cable angle
  max_angle_port_ = DeclareVectorOutputPort(
      "max_cable_angle", 1,
      &ExtendedLoadTrajectoryGenerator::CalcMaxAngle).get_index();

  // Individual outputs for compatibility
  position_port_ = DeclareVectorOutputPort(
      "position", 3,
      &ExtendedLoadTrajectoryGenerator::CalcPosition).get_index();

  velocity_port_ = DeclareVectorOutputPort(
      "velocity", 3,
      &ExtendedLoadTrajectoryGenerator::CalcVelocity).get_index();

  acceleration_port_ = DeclareVectorOutputPort(
      "acceleration", 3,
      &ExtendedLoadTrajectoryGenerator::CalcAcceleration).get_index();

  jerk_port_ = DeclareVectorOutputPort(
      "jerk", 3,
      &ExtendedLoadTrajectoryGenerator::CalcJerk).get_index();
}

ExtendedLoadTrajectoryGenerator ExtendedLoadTrajectoryGenerator::CreateAltitudeRamp(
    const Eigen::Vector3d& initial_pos,
    const Eigen::Vector3d& final_pos,
    double start_time,
    const quad_rope_lift::LoadTrajectoryParams& traj_params,
    const FeasibilityParams& feas_params) {

  // Compute duration based on distance and velocity limit
  double distance = (final_pos - initial_pos).norm();
  double cruise_vel = traj_params.max_velocity;
  double accel_time = cruise_vel / traj_params.max_acceleration;
  double accel_dist = 0.5 * traj_params.max_acceleration * accel_time * accel_time;
  double cruise_dist = std::max(0.0, distance - 2 * accel_dist);
  double cruise_time = cruise_dist / cruise_vel;
  double total_move_time = 2 * accel_time + cruise_time;

  std::vector<quad_rope_lift::LoadWaypoint> waypoints;

  // Initial hold
  quad_rope_lift::LoadWaypoint wp0;
  wp0.position = initial_pos;
  wp0.arrival_time = 0.0;
  wp0.hold_time = start_time;
  waypoints.push_back(wp0);

  // Final position
  quad_rope_lift::LoadWaypoint wp1;
  wp1.position = final_pos;
  wp1.arrival_time = start_time + total_move_time;
  wp1.hold_time = 1e6;  // Hold indefinitely
  waypoints.push_back(wp1);

  return ExtendedLoadTrajectoryGenerator(waypoints, traj_params, feas_params);
}

ExtendedLoadTrajectoryGenerator ExtendedLoadTrajectoryGenerator::CreateHover(
    const Eigen::Vector3d& hover_position,
    const FeasibilityParams& feas_params) {

  std::vector<quad_rope_lift::LoadWaypoint> waypoints;
  quad_rope_lift::LoadWaypoint wp;
  wp.position = hover_position;
  wp.arrival_time = 0.0;
  wp.hold_time = 1e9;
  waypoints.push_back(wp);

  return ExtendedLoadTrajectoryGenerator(waypoints, quad_rope_lift::LoadTrajectoryParams{}, feas_params);
}

void ExtendedLoadTrajectoryGenerator::EvaluateTrajectoryWithJerk(
    double t,
    Eigen::Vector3d& position,
    Eigen::Vector3d& velocity,
    Eigen::Vector3d& acceleration,
    Eigen::Vector3d& jerk) const {

  // Get base trajectory values at time t
  // We need to access the base generator's internal evaluation
  // Since we don't have direct access, we create a temporary context
  auto base_context = base_generator_->CreateDefaultContext();
  base_context->SetTime(t);

  // Evaluate position, velocity, acceleration
  position = base_generator_->get_position_output_port().Eval(*base_context);
  velocity = base_generator_->get_velocity_output_port().Eval(*base_context);
  acceleration = base_generator_->get_acceleration_output_port().Eval(*base_context);

  // Compute jerk via numerical differentiation
  double t_plus = t + kJerkDeltaT;
  double t_minus = std::max(0.0, t - kJerkDeltaT);

  base_context->SetTime(t_plus);
  Eigen::Vector3d accel_plus = base_generator_->get_acceleration_output_port().Eval(*base_context);

  base_context->SetTime(t_minus);
  Eigen::Vector3d accel_minus = base_generator_->get_acceleration_output_port().Eval(*base_context);

  // Central difference for jerk
  jerk = (accel_plus - accel_minus) / (t_plus - t_minus);
}

double ExtendedLoadTrajectoryGenerator::ComputeRequiredCableAngle(
    const Eigen::Vector3d& acceleration) const {
  // From force balance on load:
  // Sum of cable forces = m_L * (a + g*e3)
  //
  // For horizontal motion, cables must tilt. The required cable angle
  // from vertical satisfies:
  //   tan(θ) = a_horizontal / (g + a_z)
  //
  // This is Equation (9) feasibility condition:
  //   ||a_L_horiz|| < g * tan(α_max)

  double a_horiz = std::sqrt(acceleration.x() * acceleration.x() +
                             acceleration.y() * acceleration.y());
  double a_vert = feas_params_.gravity + acceleration.z();

  if (a_vert <= 0.0) {
    // Invalid - can't accelerate downward faster than gravity
    return M_PI / 2.0;  // Return 90 degrees (infeasible)
  }

  return std::atan2(a_horiz, a_vert);
}

void ExtendedLoadTrajectoryGenerator::CalcFullTrajectory(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d pos, vel, accel, jerk;
  EvaluateTrajectoryWithJerk(context.get_time(), pos, vel, accel, jerk);

  Eigen::VectorXd out(12);
  out.segment<3>(0) = pos;
  out.segment<3>(3) = vel;
  out.segment<3>(6) = accel;
  out.segment<3>(9) = jerk;
  output->get_mutable_value() = out;
}

void ExtendedLoadTrajectoryGenerator::CalcFeasibility(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d pos, vel, accel, jerk;
  EvaluateTrajectoryWithJerk(context.get_time(), pos, vel, accel, jerk);

  double required_angle = ComputeRequiredCableAngle(accel);
  double feasible = (required_angle < feas_params_.max_cable_angle) ? 1.0 : 0.0;

  output->get_mutable_value() << feasible;
}

void ExtendedLoadTrajectoryGenerator::CalcMaxAngle(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d pos, vel, accel, jerk;
  EvaluateTrajectoryWithJerk(context.get_time(), pos, vel, accel, jerk);

  double required_angle = ComputeRequiredCableAngle(accel);
  output->get_mutable_value() << required_angle;
}

void ExtendedLoadTrajectoryGenerator::CalcPosition(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  auto base_context = base_generator_->CreateDefaultContext();
  base_context->SetTime(context.get_time());
  Eigen::Vector3d pos = base_generator_->get_position_output_port().Eval(*base_context);
  output->get_mutable_value() = pos;
}

void ExtendedLoadTrajectoryGenerator::CalcVelocity(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  auto base_context = base_generator_->CreateDefaultContext();
  base_context->SetTime(context.get_time());
  Eigen::Vector3d vel = base_generator_->get_velocity_output_port().Eval(*base_context);
  output->get_mutable_value() = vel;
}

void ExtendedLoadTrajectoryGenerator::CalcAcceleration(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  auto base_context = base_generator_->CreateDefaultContext();
  base_context->SetTime(context.get_time());
  Eigen::Vector3d accel = base_generator_->get_acceleration_output_port().Eval(*base_context);
  output->get_mutable_value() = accel;
}

void ExtendedLoadTrajectoryGenerator::CalcJerk(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d pos, vel, accel, jerk;
  EvaluateTrajectoryWithJerk(context.get_time(), pos, vel, accel, jerk);
  output->get_mutable_value() = jerk;
}

}  // namespace tether_lift
