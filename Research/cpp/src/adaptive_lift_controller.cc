#include "adaptive_lift_controller.h"

#include <algorithm>
#include <cmath>

#include <drake/common/drake_assert.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/math/spatial_algebra.h>
#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::math::RotationMatrix;
using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::SpatialForce;
using drake::systems::BasicVector;
using drake::systems::Context;

AdaptiveLiftController::AdaptiveLiftController(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::multibody::RigidBody<double>& quadcopter_body,
    const AdaptiveLiftControllerParams& params)
    : plant_(plant),
      quad_body_index_(quadcopter_body.index()),
      params_(params) {

  // Set quadcopter mass from body
  params_.quadcopter_mass = quadcopter_body.default_mass();

  // Input ports
  // Quad state: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz] = 13
  quad_state_port_ =
      DeclareVectorInputPort("quad_state", BasicVector<double>(13))
          .get_index();

  cable_tension_port_ =
      DeclareVectorInputPort("cable_tension", BasicVector<double>(1))
          .get_index();

  cable_direction_port_ =
      DeclareVectorInputPort("cable_direction", BasicVector<double>(3))
          .get_index();

  theta_hat_port_ =
      DeclareVectorInputPort("theta_hat", BasicVector<double>(1))
          .get_index();

  load_position_est_port_ =
      DeclareVectorInputPort("load_position_est", BasicVector<double>(3))
          .get_index();

  load_velocity_est_port_ =
      DeclareVectorInputPort("load_velocity_est", BasicVector<double>(3))
          .get_index();

  // Desired load trajectory: [px,py,pz, vx,vy,vz] = 6
  load_trajectory_des_port_ =
      DeclareVectorInputPort("load_trajectory_des", BasicVector<double>(6))
          .get_index();

  // Output port
  control_port_ = DeclareAbstractOutputPort(
      "control_force",
      &AdaptiveLiftController::CalcControlForce)
      .get_index();
}

void AdaptiveLiftController::set_mass(double mass) {
  params_.quadcopter_mass = mass;
}

void AdaptiveLiftController::ExtractQuadState(
    const Eigen::VectorXd& state,
    Eigen::Vector3d& position,
    Eigen::Quaterniond& orientation,
    Eigen::Vector3d& velocity,
    Eigen::Vector3d& angular_velocity) const {

  position = state.segment<3>(0);
  orientation = Eigen::Quaterniond(state[3], state[4], state[5], state[6]);
  orientation.normalize();
  velocity = state.segment<3>(7);
  angular_velocity = state.segment<3>(10);
}

void AdaptiveLiftController::ComputeLoadTrajectory(
    double t,
    Eigen::Vector3d& pos_des,
    Eigen::Vector3d& vel_des) const {

  // Simple altitude ramp trajectory for the load
  double altitude;
  double vz_des = 0.0;

  if (t < params_.ascent_start_time) {
    // Hold at initial altitude
    altitude = params_.initial_altitude;
  } else {
    double ascent_time = t - params_.ascent_start_time;
    double target_duration = (params_.final_altitude - params_.initial_altitude) /
                             params_.climb_rate;

    if (ascent_time < target_duration) {
      // Ascending
      altitude = params_.initial_altitude + params_.climb_rate * ascent_time;
      vz_des = params_.climb_rate;
    } else {
      // Reached final altitude
      altitude = params_.final_altitude;
    }
  }

  pos_des = Eigen::Vector3d(0, 0, altitude);
  vel_des = Eigen::Vector3d(0, 0, vz_des);
}

void AdaptiveLiftController::CalcControlForce(
    const Context<double>& context,
    std::vector<ExternallyAppliedSpatialForce<double>>* output) const {

  output->clear();

  // === Get inputs ===
  const auto& quad_state_raw = get_input_port(quad_state_port_).Eval(context);
  const double T_cable = get_input_port(cable_tension_port_).Eval(context)[0];
  const auto& n_raw = get_input_port(cable_direction_port_).Eval(context);
  const double theta_hat = get_input_port(theta_hat_port_).Eval(context)[0];
  const auto& p_L_est_raw = get_input_port(load_position_est_port_).Eval(context);
  const auto& v_L_est_raw = get_input_port(load_velocity_est_port_).Eval(context);
  const auto& traj_des_raw = get_input_port(load_trajectory_des_port_).Eval(context);

  // Extract quad state
  Eigen::Vector3d p_Q, v_Q, omega;
  Eigen::Quaterniond q_Q;
  ExtractQuadState(Eigen::Map<const Eigen::VectorXd>(quad_state_raw.data(), 13),
                   p_Q, q_Q, v_Q, omega);

  // Cable direction (normalized)
  Eigen::Vector3d n(n_raw[0], n_raw[1], n_raw[2]);
  double n_norm = n.norm();
  if (n_norm > 1e-6) {
    n /= n_norm;
  } else {
    n = Eigen::Vector3d(0, 0, -1);  // Default: straight down
  }

  // Estimated load state
  Eigen::Vector3d p_L_est(p_L_est_raw[0], p_L_est_raw[1], p_L_est_raw[2]);
  Eigen::Vector3d v_L_est(v_L_est_raw[0], v_L_est_raw[1], v_L_est_raw[2]);

  // Desired load trajectory
  Eigen::Vector3d p_L_des(traj_des_raw[0], traj_des_raw[1], traj_des_raw[2]);
  Eigen::Vector3d v_L_des(traj_des_raw[3], traj_des_raw[4], traj_des_raw[5]);

  // If trajectory input is zero, use internal trajectory generator
  if (p_L_des.norm() < 1e-6 && v_L_des.norm() < 1e-6) {
    ComputeLoadTrajectory(context.get_time(), p_L_des, v_L_des);
  }

  // === Compute desired quadcopter position from load-centric tracking ===
  // Quad should be at: p_Q_des = p_L_des + formation_offset
  Eigen::Vector3d p_Q_des = p_L_des + params_.formation_offset;
  Eigen::Vector3d v_Q_des = v_L_des;  // Quad velocity tracks load velocity

  // === Position control ===
  // Error from desired quad position
  Eigen::Vector3d e_pos = p_Q - p_Q_des;
  Eigen::Vector3d e_vel = v_Q - v_Q_des;

  // PD control for horizontal acceleration
  double ax_des = -params_.position_kp * e_pos.x() - params_.position_kd * e_vel.x();
  double ay_des = -params_.position_kp * e_pos.y() - params_.position_kd * e_vel.y();

  // PD control for vertical acceleration
  double az_des = -params_.altitude_kp * e_pos.z() - params_.altitude_kd * e_vel.z();

  // === Load tracking feedback ===
  // Additional term to help load track its trajectory
  Eigen::Vector3d e_L = p_L_est - p_L_des;
  Eigen::Vector3d e_L_dot = v_L_est - v_L_des;

  // Load tracking adds to desired acceleration (pulls quad to help load)
  ax_des -= params_.load_track_kp * e_L.x() + params_.load_track_kd * e_L_dot.x();
  ay_des -= params_.load_track_kp * e_L.y() + params_.load_track_kd * e_L_dot.y();
  az_des -= params_.load_track_kp * e_L.z() + params_.load_track_kd * e_L_dot.z();

  // === Adaptive load feedforward ===
  // θ̂ * g is the estimated weight share that this quad must support
  double load_feedforward = 0.0;
  if (theta_hat > params_.theta_min_for_feedforward) {
    load_feedforward = params_.theta_feedforward_gain * theta_hat * kGravity;
  }

  // === Tension regulation ===
  // Keep cable at nominal tension to ensure controllability
  double tension_error = T_cable - params_.nominal_cable_tension;
  double tension_adjustment = 0.0;

  if (std::abs(tension_error) > params_.tension_deadband) {
    // Outside deadband: regulate tension
    double signed_error = tension_error - std::copysign(params_.tension_deadband, tension_error);
    tension_adjustment = -params_.tension_kp * signed_error;
  }

  // === Convert to desired attitude and thrust ===
  // Desired tilt angles from horizontal acceleration
  double pitch_des = std::clamp(ax_des / kGravity, -params_.max_tilt_angle, params_.max_tilt_angle);
  double roll_des = std::clamp(-ay_des / kGravity, -params_.max_tilt_angle, params_.max_tilt_angle);

  // Total vertical thrust needed
  // = quad weight + commanded z accel + load share feedforward + tension regulation
  double thrust_z = params_.quadcopter_mass * (kGravity + az_des) +
                    load_feedforward + tension_adjustment;

  // If cable is under tension, add component along cable direction
  // This helps maintain tension and aligns thrust with load direction
  double cable_alignment_thrust = 0.0;
  if (T_cable > params_.min_cable_tension_for_alignment) {
    // Add thrust component toward load to maintain cable tension
    // The cable direction n points from quad toward load
    double vertical_component = -n.z();  // How much of cable points down
    if (vertical_component > 0.1) {
      cable_alignment_thrust = params_.cable_alignment_gain *
                               (1.0 - vertical_component) * load_feedforward;
    }
  }

  double total_thrust = thrust_z + cable_alignment_thrust;
  total_thrust = std::clamp(total_thrust, params_.min_thrust, params_.max_thrust);

  // === Attitude control ===
  // Get current Euler angles from quaternion
  RotationMatrix<double> R(q_Q);
  // Roll-pitch-yaw extraction (intrinsic XYZ convention)
  Eigen::Vector3d rpy = R.IsValid() ? R.ToRollPitchYaw().vector() : Eigen::Vector3d::Zero();

  double roll_current = rpy[0];
  double pitch_current = rpy[1];
  double yaw_current = rpy[2];

  double roll_error = roll_des - roll_current;
  double pitch_error = pitch_des - pitch_current;
  double yaw_error = 0.0 - yaw_current;  // Maintain zero yaw

  // PD attitude control
  double tau_roll = params_.attitude_kp * roll_error - params_.attitude_kd * omega.x();
  double tau_pitch = params_.attitude_kp * pitch_error - params_.attitude_kd * omega.y();
  double tau_yaw = 0.5 * params_.attitude_kp * yaw_error - params_.attitude_kd * omega.z();

  // Clamp torques
  tau_roll = std::clamp(tau_roll, -params_.max_torque, params_.max_torque);
  tau_pitch = std::clamp(tau_pitch, -params_.max_torque, params_.max_torque);
  tau_yaw = std::clamp(tau_yaw, -params_.max_torque, params_.max_torque);

  // === Apply forces ===
  // Thrust acts in body z-direction
  Eigen::Vector3d thrust_body(0, 0, total_thrust);
  Eigen::Vector3d thrust_world = R.matrix() * thrust_body;

  // Create spatial force
  Eigen::Vector3d torque_body(tau_roll, tau_pitch, tau_yaw);

  SpatialForce<double> spatial_force(torque_body, thrust_world);

  ExternallyAppliedSpatialForce<double> applied_force;
  applied_force.body_index = quad_body_index_;
  applied_force.p_BoBq_B = Eigen::Vector3d::Zero();  // Force at body origin
  applied_force.F_Bq_W = spatial_force;

  output->push_back(applied_force);
}

}  // namespace quad_rope_lift
