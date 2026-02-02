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
      initial_altitude_(params.initial_altitude),
      final_altitude_(params.final_altitude),
      ascent_start_time_(params.ascent_start_time),
      climb_rate_(params.climb_rate),
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

  // Pre-compute ascent duration
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

  // Compute desired altitude trajectory
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
      desired_altitude += altitude_adjustment;
    }
  }

  // === Altitude Controller (PD) ===
  const Eigen::Vector3d translation = pose_world.translation();
  const Eigen::Vector3d translational_vel = velocity_world.translational();

  const double altitude_error = desired_altitude - translation[2];
  const double velocity_error = desired_velocity_z - translational_vel[2];

  const double commanded_accel_z = altitude_kp_ * altitude_error + altitude_kd_ * velocity_error;
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

  // Attitude error from skew-symmetric part (inlined)
  const double attitude_error_x = 0.5 * (R(2, 1) - R(1, 2));
  const double attitude_error_y = 0.5 * (R(0, 2) - R(2, 0));
  const double attitude_error_z = 0.5 * (R(1, 0) - R(0, 1));

  // Angular velocity in body frame: omega_B = R^T * omega_W
  const Eigen::Vector3d omega_W = velocity_world.rotational();
  const double omega_Bx = R(0, 0) * omega_W[0] + R(1, 0) * omega_W[1] + R(2, 0) * omega_W[2];
  const double omega_By = R(0, 1) * omega_W[0] + R(1, 1) * omega_W[1] + R(2, 1) * omega_W[2];
  const double omega_Bz = R(0, 2) * omega_W[0] + R(1, 2) * omega_W[1] + R(2, 2) * omega_W[2];

  // PD control for torque
  double tau_x = -attitude_kp_ * attitude_error_x - attitude_kd_ * omega_Bx;
  double tau_y = -attitude_kp_ * attitude_error_y - attitude_kd_ * omega_By;
  double tau_z = -attitude_kp_ * attitude_error_z - attitude_kd_ * omega_Bz;

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
