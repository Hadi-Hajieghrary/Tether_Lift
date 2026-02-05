#pragma once

/// @file gpac_quadcopter_controller.h
/// @brief GPAC-enhanced quadcopter controller with geometric attitude control.
///
/// Upgrades the original QuadcopterLiftController with:
/// 1. Geometric SO(3) attitude control [Layer 2]
/// 2. Extended State Observer integration [Layer 4]
/// 3. Anti-swing cable direction control [Layer 1]
/// 4. Disturbance feedforward compensation

#include <Eigen/Core>
#include <optional>
#include <vector>
#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/leaf_system.h>

#include "gpac_math.h"
#include "extended_state_observer.h"

namespace quad_rope_lift {

/// Trajectory waypoint
struct GPACWaypoint {
  Eigen::Vector3d position{0, 0, 1};
  double arrival_time = 0.0;
  double hold_time = 0.0;
};

/// GPAC controller parameters
struct GPACParams {
  // Formation offset
  Eigen::Vector3d formation_offset{0, 0, 0};

  // Trajectory (legacy mode if empty)
  std::vector<GPACWaypoint> waypoints;
  double initial_altitude = 1.0;
  double final_altitude = 3.0;
  double ascent_start_time = 0.5;
  double climb_rate = 0.4;

  // Layer 1: Position control (50 Hz effective)
  double position_kp = 6.0;           ///< Position error gain
  double position_kd = 8.0;           ///< Velocity error gain
  double position_ki = 0.1;           ///< Integral gain (small)
  double max_integral = 2.0;          ///< Anti-windup limit

  // Layer 1: Anti-swing (S² control)
  double cable_kq = 4.0;              ///< Cable direction error gain
  double cable_kw = 2.0;              ///< Cable angular velocity gain
  bool enable_antiswing = true;       ///< Enable anti-swing control

  // Layer 2: Geometric attitude control (200 Hz effective)
  double attitude_kR = 8.0;           ///< Rotation error gain [Eq. 12]
  double attitude_kOmega = 1.5;       ///< Angular velocity error gain [Eq. 12]

  // Layer 4: ESO parameters
  double eso_omega = 50.0;            ///< ESO bandwidth [rad/s]
  double eso_b0 = 0.67;               ///< ESO input gain (1/mass)
  bool enable_eso_feedforward = true; ///< Enable disturbance feedforward

  // Tension-aware pickup
  double tension_kp = 0.5;
  double tension_altitude_gain = 0.003;
  double tension_altitude_max = 0.5;
  double pickup_ramp_duration = 2.0;
  double pickup_target_tension = 20.0;
  double pickup_detection_threshold = 1.0;

  // Actuator limits
  double min_thrust = 0.0;
  double max_thrust = 150.0;
  double max_torque = 10.0;

  // Physical constants
  double gravity = 9.81;
  double mass = 1.5;  ///< Quadcopter mass [kg]
};

/// @brief GPAC-enhanced quadcopter controller
///
/// Implements hierarchical control from GPAC architecture:
/// - Layer 1: Position + Anti-swing (50 Hz) → desired thrust vector
/// - Layer 2: Geometric attitude (200 Hz) → control torques
/// - Layer 4: ESO (500 Hz) → disturbance estimates
class GPACQuadcopterController final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GPACQuadcopterController);

  /// Construct the GPAC controller
  GPACQuadcopterController(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::multibody::RigidBody<double>& quadcopter_body,
      const GPACParams& params = GPACParams());

  // === Input ports ===

  /// Plant state [positions; velocities]
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Rope tension [T, fx, fy, fz]
  const drake::systems::InputPort<double>& get_tension_input_port() const {
    return get_input_port(tension_port_);
  }

  /// Cable direction [qx, qy, qz] (unit vector from load to drone)
  const drake::systems::InputPort<double>& get_cable_direction_input_port() const {
    return get_input_port(cable_direction_port_);
  }

  /// External estimated state [px, py, pz, vx, vy, vz] (optional)
  const drake::systems::InputPort<double>& get_estimated_state_input_port() const {
    return get_input_port(estimated_state_port_);
  }

  /// Estimated disturbance [dx, dy, dz] (from external ESO, optional)
  const drake::systems::InputPort<double>& get_disturbance_input_port() const {
    return get_input_port(disturbance_port_);
  }

  // === Output ports ===

  /// Control forces (spatial force)
  const drake::systems::OutputPort<double>& get_control_output_port() const {
    return get_output_port(control_port_);
  }

  /// Control vector [tau_x, tau_y, tau_z, f_x, f_y, f_z] for logging
  const drake::systems::OutputPort<double>& get_control_vector_output_port() const {
    return get_output_port(control_vector_port_);
  }

  /// Desired attitude quaternion [w, x, y, z] for logging
  const drake::systems::OutputPort<double>& get_desired_attitude_output_port() const {
    return get_output_port(desired_attitude_port_);
  }

  /// Attitude error [eR_x, eR_y, eR_z] for logging
  const drake::systems::OutputPort<double>& get_attitude_error_output_port() const {
    return get_output_port(attitude_error_port_);
  }

  // === Parameter access ===

  void set_mass(double mass) { params_.mass = mass; }
  GPACParams& mutable_params() { return params_; }
  const GPACParams& params() const { return params_; }

 private:
  void DoCalcTimeDerivatives(
      const drake::systems::Context<double>& context,
      drake::systems::ContinuousState<double>* derivatives) const override;

  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  void CalcControlForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  void CalcControlVector(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredAttitude(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcAttitudeError(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  /// Compute trajectory at time t
  void ComputeTrajectory(double t,
                         Eigen::Vector3d& pos_des,
                         Eigen::Vector3d& vel_des,
                         Eigen::Vector3d& acc_des) const;

  /// Compute Layer 1: Position + Anti-swing control
  /// Returns desired thrust vector in world frame
  Eigen::Vector3d ComputeLayer1Control(
      const drake::systems::Context<double>& context,
      const Eigen::Vector3d& position,
      const Eigen::Vector3d& velocity,
      double measured_tension) const;

  /// Compute Layer 2: Geometric attitude control
  /// Returns torque in body frame
  Eigen::Vector3d ComputeLayer2Control(
      const Eigen::Matrix3d& R,
      const Eigen::Matrix3d& R_d,
      const Eigen::Vector3d& Omega,
      const Eigen::Vector3d& Omega_d) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::multibody::BodyIndex quad_body_index_;

  // Parameters
  GPACParams params_;

  // Precomputed trajectory values
  double ascent_duration_;
  double ascent_direction_;

  // Mutable state
  mutable std::optional<double> pickup_start_time_;

  // Internal ESO (if external not connected)
  mutable gpac::DroneESO internal_eso_;
  mutable double last_update_time_ = -1.0;
  mutable Eigen::Vector3d last_control_accel_ = Eigen::Vector3d::Zero();

  // Port indices
  int plant_state_port_{};
  int tension_port_{};
  int cable_direction_port_{};
  int estimated_state_port_{};
  int disturbance_port_{};
  int control_port_{};
  int control_vector_port_{};
  int desired_attitude_port_{};
  int attitude_error_port_{};

  // Continuous state for integral control (3 elements for xyz position integral)
  // State layout: [int_x, int_y, int_z]
};

}  // namespace quad_rope_lift
