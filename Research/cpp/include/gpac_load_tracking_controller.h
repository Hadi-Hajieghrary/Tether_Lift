#pragma once

/// @file gpac_load_tracking_controller.h
/// @brief GPAC-enhanced load tracking controller with anti-swing control.
///
/// Upgrades the original LoadTrackingController with:
/// 1. S² cable direction tracking [Eq. 8-9]
/// 2. Anti-swing control with direction damping
/// 3. ESO disturbance feedforward
/// 4. Geometric desired attitude computation

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <drake/systems/framework/leaf_system.h>

#include "gpac_math.h"

namespace tether_lift {

/// @brief Parameters for GPAC load tracking controller
struct GPACLoadTrackingParams {
  // Physical parameters
  double mass = 1.5;                 ///< Drone mass [kg]
  double gravity = 9.81;             ///< Gravity [m/s²]
  double cable_length = 1.0;         ///< Nominal cable length [m]

  // Position tracking gains (Layer 1)
  Eigen::Vector3d Kp{6.0, 6.0, 8.0}; ///< Position error gain
  Eigen::Vector3d Kd{8.0, 8.0, 10.0}; ///< Velocity error gain
  Eigen::Vector3d Ki{0.1, 0.1, 0.2}; ///< Integral gain
  double max_integral = 2.0;          ///< Anti-windup limit

  // Cable direction control (S² anti-swing)
  double kq = 4.0;                   ///< Cable direction error gain [Eq. 8]
  double kw = 2.0;                   ///< Cable angular velocity gain
  bool enable_antiswing = true;       ///< Enable anti-swing control

  // Cable tension feedforward
  double cable_compensation_gain = 1.0;
  bool use_actual_tension_feedforward = true;

  // ESO disturbance feedforward
  bool enable_eso_feedforward = true;

  // Actuator limits
  double min_thrust = 0.0;
  double max_thrust = 50.0;
  double max_tilt = 0.5;             ///< Maximum tilt angle [rad]
};

/// @brief GPAC-enhanced load tracking controller
///
/// Implements GPAC Layer 1 control:
/// - Position tracking with PID
/// - Anti-swing control on S² (cable direction)
/// - ESO disturbance feedforward
class GPACLoadTrackingController final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GPACLoadTrackingController);

  explicit GPACLoadTrackingController(
      const GPACLoadTrackingParams& params = GPACLoadTrackingParams());

  // === Input port accessors ===

  /// Drone state: [position(3), quaternion(4), velocity(3), angular_velocity(3)] = 13D
  const drake::systems::InputPort<double>& get_drone_state_input() const {
    return get_input_port(drone_state_port_);
  }

  /// Drone trajectory: [p_des(3), v_des(3), a_des(3)] = 9D
  const drake::systems::InputPort<double>& get_drone_trajectory_input() const {
    return get_input_port(drone_trajectory_port_);
  }

  /// Desired tension T_des (scalar)
  const drake::systems::InputPort<double>& get_desired_tension_input() const {
    return get_input_port(desired_tension_port_);
  }

  /// Desired cable direction q_des (3D unit vector)
  const drake::systems::InputPort<double>& get_desired_cable_direction_input() const {
    return get_input_port(desired_cable_direction_port_);
  }

  /// Actual cable direction q (3D unit vector)
  const drake::systems::InputPort<double>& get_actual_cable_direction_input() const {
    return get_input_port(actual_cable_direction_port_);
  }

  /// Actual tension T (scalar)
  const drake::systems::InputPort<double>& get_actual_tension_input() const {
    return get_input_port(actual_tension_port_);
  }

  /// Cable angular velocity ω_q (3D, optional)
  const drake::systems::InputPort<double>& get_cable_angular_velocity_input() const {
    return get_input_port(cable_angular_velocity_port_);
  }

  /// ESO disturbance estimate (3D, optional)
  const drake::systems::InputPort<double>& get_disturbance_input() const {
    return get_input_port(disturbance_port_);
  }

  // === Output port accessors ===

  /// Thrust force vector in world frame (3D)
  const drake::systems::OutputPort<double>& get_thrust_force_output() const {
    return get_output_port(thrust_force_port_);
  }

  /// Thrust magnitude (scalar)
  const drake::systems::OutputPort<double>& get_thrust_magnitude_output() const {
    return get_output_port(thrust_magnitude_port_);
  }

  /// Desired attitude as quaternion [w, x, y, z]
  const drake::systems::OutputPort<double>& get_desired_attitude_output() const {
    return get_output_port(desired_attitude_port_);
  }

  /// Desired angular velocity (3D)
  const drake::systems::OutputPort<double>& get_desired_angular_velocity_output() const {
    return get_output_port(desired_angular_velocity_port_);
  }

  /// Anti-swing force component (3D, for logging)
  const drake::systems::OutputPort<double>& get_antiswing_force_output() const {
    return get_output_port(antiswing_force_port_);
  }

  /// Cable direction error [Eq. 8] (3D, for logging)
  const drake::systems::OutputPort<double>& get_cable_error_output() const {
    return get_output_port(cable_error_port_);
  }

  // === Parameter access ===

  GPACLoadTrackingParams& mutable_params() { return params_; }
  const GPACLoadTrackingParams& params() const { return params_; }

 private:
  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  void DoCalcTimeDerivatives(
      const drake::systems::Context<double>& context,
      drake::systems::ContinuousState<double>* derivatives) const override;

  /// Compute thrust vector (main control computation)
  Eigen::Vector3d ComputeThrustVector(
      const drake::systems::Context<double>& context) const;

  /// Compute anti-swing force from S² cable error
  Eigen::Vector3d ComputeAntiSwingForce(
      const drake::systems::Context<double>& context) const;

  /// Convert thrust vector to desired quaternion
  Eigen::Quaterniond ThrustVectorToQuaternion(
      const Eigen::Vector3d& thrust, double yaw_des = 0.0) const;

  // Output calculation methods
  void CalcThrustForce(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcThrustMagnitude(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredAttitude(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredAngularVelocity(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcAntiSwingForce(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcCableError(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Parameters
  GPACLoadTrackingParams params_;

  // Port indices
  int drone_state_port_{};
  int drone_trajectory_port_{};
  int desired_tension_port_{};
  int desired_cable_direction_port_{};
  int actual_cable_direction_port_{};
  int actual_tension_port_{};
  int cable_angular_velocity_port_{};
  int disturbance_port_{};

  int thrust_force_port_{};
  int thrust_magnitude_port_{};
  int desired_attitude_port_{};
  int desired_angular_velocity_port_{};
  int antiswing_force_port_{};
  int cable_error_port_{};
};

}  // namespace tether_lift
