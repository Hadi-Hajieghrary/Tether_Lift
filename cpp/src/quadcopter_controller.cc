#include "quadcopter_controller.h"

#include <algorithm>
#include <cmath>

#include <drake/multibody/math/spatial_force.h>

namespace quad_rope_lift {

using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::multibody::SpatialForce;
using drake::systems::BasicVector;
using drake::systems::Context;

QuadcopterLiftController::QuadcopterLiftController(
    const MultibodyPlant<double>& plant,
    const RigidBody<double>& quadcopter_body,
    const ControllerParams& params)
    : plant_(plant),
      quad_body_index_(quadcopter_body.index()),
      mass_(quadcopter_body.default_mass()),
      formation_offset_(params.formation_offset),
      waypoints_(params.waypoints),
      use_waypoints_(!params.waypoints.empty()),
      initial_altitude_(params.initial_altitude),
      final_altitude_(params.final_altitude),
      ascent_start_time_(params.ascent_start_time),
      climb_rate_(params.climb_rate),
      position_kp_(params.position_kp),
      position_kd_(params.position_kd),
      max_tilt_angle_(params.max_tilt_angle),
      altitude_kp_(params.altitude_kp),
      altitude_kd_(params.altitude_kd),
      attitude_kp_(params.attitude_kp),
      attitude_kd_(params.attitude_kd),
      tension_kp_(params.tension_feedback_kp),
      tension_altitude_gain_(params.tension_altitude_gain),
      tension_altitude_max_(params.tension_altitude_max),
      pickup_duration_(params.pickup_ramp_duration),
      pickup_target_tension_(params.pickup_target_tension),
      pickup_threshold_(params.pickup_detection_threshold),
      min_thrust_(params.min_thrust),
      max_thrust_(params.max_thrust),
      max_torque_(params.max_torque),
      gravity_(params.gravity) {

  // Pre-compute ascent duration (for legacy trajectory mode)
  const double altitude_change = final_altitude_ - initial_altitude_;
  ascent_duration_ = (climb_rate_ > 0.0) ?
      std::abs(altitude_change) / climb_rate_ : 0.0;
  ascent_direction_ = (altitude_change >= 0.0) ? 1.0 : -1.0;

  // Declare input ports
  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  tension_port_ = DeclareVectorInputPort(
      "rope_tension",
      BasicVector<double>(4))
      .get_index();

  // Declare output port for control forces
  control_port_ = DeclareAbstractOutputPort(
      "control_force",
      &QuadcopterLiftController::CalcControlForce)
      .get_index();
}

void QuadcopterLiftController::ComputeTrajectory(
    double t, Eigen::Vector3d& pos_des, Eigen::Vector3d& vel_des) const {

  if (use_waypoints_ && !waypoints_.empty()) {
    // Waypoint-based trajectory with linear interpolation
    // Find current segment
    double segment_start_time = 0.0;
    for (size_t i = 0; i < waypoints_.size(); ++i) {
      const auto& wp = waypoints_[i];
      const double segment_end_time = wp.arrival_time;
      const double hold_end_time = segment_end_time + wp.hold_time;

      if (t <= segment_end_time) {
        // In transit to waypoint i
        if (i == 0) {
          // First waypoint: already at initial position
          pos_des = wp.position + formation_offset_;
          vel_des.setZero();
        } else {
          // Interpolate from previous waypoint
          const auto& prev_wp = waypoints_[i - 1];
          const double segment_duration = segment_end_time - segment_start_time;
          if (segment_duration > 1e-6) {
            const double alpha = (t - segment_start_time) / segment_duration;
            pos_des = (1.0 - alpha) * prev_wp.position + alpha * wp.position + formation_offset_;
            vel_des = (wp.position - prev_wp.position) / segment_duration;
          } else {
            pos_des = wp.position + formation_offset_;
            vel_des.setZero();
          }
        }
        return;
      } else if (t <= hold_end_time) {
        // Holding at waypoint i
        pos_des = wp.position + formation_offset_;
        vel_des.setZero();
        return;
      }
      segment_start_time = hold_end_time;
    }

    // Past all waypoints: hold at final waypoint
    pos_des = waypoints_.back().position + formation_offset_;
    vel_des.setZero();

  } else {
    // Legacy altitude-only trajectory (for backward compatibility)
    double desired_altitude;
    double desired_velocity_z;

    if (t <= ascent_start_time_ || climb_rate_ <= 0.0) {
      desired_altitude = initial_altitude_;
      desired_velocity_z = 0.0;
    } else if (t >= ascent_start_time_ + ascent_duration_) {
      desired_altitude = final_altitude_;
      desired_velocity_z = 0.0;
    } else {
      const double elapsed = t - ascent_start_time_;
      desired_altitude = initial_altitude_ + ascent_direction_ * climb_rate_ * elapsed;
      desired_velocity_z = ascent_direction_ * climb_rate_;
    }

    pos_des = Eigen::Vector3d(formation_offset_.x(), formation_offset_.y(), desired_altitude);
    vel_des = Eigen::Vector3d(0.0, 0.0, desired_velocity_z);
  }
}

void QuadcopterLiftController::CalcControlForce(
    const Context<double>& context,
    std::vector<ExternallyAppliedSpatialForce<double>>* output) const {

  // Read inputs
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);
  const auto& tension_data = get_input_port(tension_port_).Eval(context);
  const double measured_tension = tension_data[0];

  // Create plant context and set state
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  const double t = context.get_time();

  // Detect start of pickup phase (rope becomes taut)
  if (!pickup_start_time_.has_value() && measured_tension >= pickup_threshold_) {
    pickup_start_time_ = t;
  }

  // Get quadcopter pose and velocity
  const auto& quad_body = plant_.get_body(quad_body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, quad_body);
  const auto& velocity_world = plant_.EvalBodySpatialVelocityInWorld(*plant_context, quad_body);

  // Compute desired trajectory
  Eigen::Vector3d pos_des, vel_des;
  ComputeTrajectory(t, pos_des, vel_des);

  // === Tension-Aware Pickup Logic ===
  bool in_pickup_phase = false;
  double target_tension = pickup_target_tension_;

  if (pickup_start_time_.has_value()) {
    const double time_since_pickup = t - pickup_start_time_.value();

    if (time_since_pickup >= 0.0 && time_since_pickup <= pickup_duration_) {
      in_pickup_phase = true;
      double ramp_fraction = time_since_pickup / pickup_duration_;
      ramp_fraction = std::clamp(ramp_fraction, 0.0, 1.0);
      target_tension = ramp_fraction * pickup_target_tension_;

      const double tension_error = target_tension - measured_tension;
      double altitude_adjustment = tension_altitude_gain_ * tension_error;
      altitude_adjustment = std::clamp(altitude_adjustment,
                                       -tension_altitude_max_,
                                       tension_altitude_max_);
      pos_des.z() += altitude_adjustment;
    }
  }

  // === Position Controller (PD) ===
  const Eigen::Vector3d translation = pose_world.translation();
  const Eigen::Vector3d translational_vel = velocity_world.translational();

  // Position and velocity errors
  const Eigen::Vector3d pos_error = pos_des - translation;
  const Eigen::Vector3d vel_error = vel_des - translational_vel;

  // X/Y position control -> desired tilt angles
  // Desired acceleration in x/y
  const double ax_des = position_kp_ * pos_error.x() + position_kd_ * vel_error.x();
  const double ay_des = position_kp_ * pos_error.y() + position_kd_ * vel_error.y();

  // Convert to desired tilt angles (small angle approximation)
  // For a quadcopter: pitch forward (positive pitch) -> move in +x
  //                   roll right (positive roll) -> move in +y
  double pitch_des = std::clamp(ax_des / gravity_, -max_tilt_angle_, max_tilt_angle_);
  double roll_des = std::clamp(-ay_des / gravity_, -max_tilt_angle_, max_tilt_angle_);

  // Altitude control (z)
  const double commanded_accel_z = altitude_kp_ * pos_error.z() + altitude_kd_ * vel_error.z();
  double thrust = mass_ * (gravity_ + commanded_accel_z);

  // Add tension feedforward and feedback
  const double tension_feedforward_limit = in_pickup_phase ?
      target_tension : pickup_target_tension_;
  const double tension_ff = std::clamp(measured_tension, 0.0, tension_feedforward_limit);
  thrust += tension_ff;

  if (in_pickup_phase) {
    thrust += tension_kp_ * (target_tension - measured_tension);
  }

  // Saturate thrust
  thrust = std::clamp(thrust, min_thrust_, max_thrust_);

  // === Attitude Controller (PD) ===
  const Eigen::Matrix3d R = pose_world.rotation().matrix();

  // Current attitude errors (from identity)
  // Using small-angle approximation for roll/pitch from rotation matrix
  const double current_roll = std::atan2(R(2, 1), R(2, 2));
  const double current_pitch = std::asin(-R(2, 0));
  const double current_yaw_error = 0.5 * (R(1, 0) - R(0, 1));  // Keep yaw at zero

  // Attitude errors (tracking desired roll/pitch)
  const double roll_error = roll_des - current_roll;
  const double pitch_error = pitch_des - current_pitch;
  const double yaw_error = -current_yaw_error;  // Stabilize to zero yaw

  // Angular velocity in body frame: omega_B = R^T * omega_W
  const Eigen::Vector3d omega_W = velocity_world.rotational();
  const double omega_Bx = R(0, 0) * omega_W[0] + R(1, 0) * omega_W[1] + R(2, 0) * omega_W[2];
  const double omega_By = R(0, 1) * omega_W[0] + R(1, 1) * omega_W[1] + R(2, 1) * omega_W[2];
  const double omega_Bz = R(0, 2) * omega_W[0] + R(1, 2) * omega_W[1] + R(2, 2) * omega_W[2];

  // PD control for torque
  double tau_x = attitude_kp_ * roll_error - attitude_kd_ * omega_Bx;
  double tau_y = attitude_kp_ * pitch_error - attitude_kd_ * omega_By;
  double tau_z = attitude_kp_ * yaw_error - attitude_kd_ * omega_Bz;

  // Saturate torque
  tau_x = std::clamp(tau_x, -max_torque_, max_torque_);
  tau_y = std::clamp(tau_y, -max_torque_, max_torque_);
  tau_z = std::clamp(tau_z, -max_torque_, max_torque_);

  // Transform torque to world frame: torque_W = R * torque_B
  const Eigen::Vector3d torque_body(tau_x, tau_y, tau_z);
  const Eigen::Vector3d torque_world = R * torque_body;

  // Thrust in body z-direction, transformed to world frame
  const Eigen::Vector3d force_world = R.col(2) * thrust;

  // Package output
  output->clear();
  output->reserve(1);

  ExternallyAppliedSpatialForce<double> control_wrench;
  control_wrench.body_index = quad_body_index_;
  control_wrench.p_BoBq_B = Eigen::Vector3d::Zero();
  control_wrench.F_Bq_W = SpatialForce<double>(torque_world, force_world);
  output->push_back(control_wrench);
}

}  // namespace quad_rope_lift
