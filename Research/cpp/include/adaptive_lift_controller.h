#pragma once

/// @file adaptive_lift_controller.h
/// @brief N-independent adaptive controller for cooperative payload lift.
///
/// This controller implements decentralized control where each quadcopter
/// operates independently without knowledge of N (total number of drones)
/// or m_L (load mass). Instead, it uses an adaptive estimate θ̂ ≈ m_L/N
/// from the AdaptiveLoadEstimator.
///
/// Key features:
///   1. N-independent: Controller law uses only θ̂, not N or m_L
///   2. Load-centric tracking: Tracks desired load trajectory
///   3. Tension-based coupling: Uses cable force feedback
///   4. Adaptive feedforward: θ̂*g for gravity compensation

#include <Eigen/Core>
#include <optional>
#include <vector>

#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the adaptive lift controller.
struct AdaptiveLiftControllerParams {
  // Formation offset relative to load's desired position
  Eigen::Vector3d formation_offset{0, 0, 1.0};  ///< Offset from load to quad [m]

  // Trajectory parameters (used if waypoints empty)
  double initial_altitude = 1.0;       ///< Initial load hover height [m]
  double final_altitude = 3.0;         ///< Target load altitude [m]
  double ascent_start_time = 2.0;      ///< Time to begin lifting [s]
  double climb_rate = 0.3;             ///< Vertical climb rate [m/s]

  // Load tracking gains (controls how quad tracks load-referenced position)
  double load_track_kp = 8.0;          ///< P gain for load position tracking
  double load_track_kd = 5.0;          ///< D gain for load velocity tracking

  // Position control gains (for own position regulation)
  double position_kp = 12.0;           ///< P gain for position error
  double position_kd = 7.0;            ///< D gain for velocity error
  double max_tilt_angle = 0.35;        ///< Max tilt for x/y control [rad] (~20 deg)

  // Altitude control gains
  double altitude_kp = 18.0;           ///< P gain for altitude
  double altitude_kd = 10.0;           ///< D gain for vertical velocity

  // Attitude control gains
  double attitude_kp = 10.0;           ///< P gain for attitude error
  double attitude_kd = 2.0;            ///< D gain for angular velocity

  // Tension control parameters
  double tension_kp = 0.3;             ///< Gain for tension error to thrust
  double tension_kd = 0.05;            ///< D gain for tension rate
  double nominal_cable_tension = 15.0; ///< Expected quasi-static tension [N]
  double tension_deadband = 2.0;       ///< Ignore tension error within this band [N]

  // Adaptive feedforward parameters
  double theta_feedforward_gain = 1.0; ///< Multiplier for θ̂*g feedforward
  double theta_min_for_feedforward = 0.5; ///< Min θ̂ to enable feedforward [kg]

  // Cable direction coupling
  double cable_alignment_gain = 2.0;   ///< Gain for aligning thrust with cable
  double min_cable_tension_for_alignment = 5.0;  ///< Min tension to use cable direction [N]

  // Actuator limits
  double min_thrust = 0.0;             ///< Minimum thrust [N]
  double max_thrust = 180.0;           ///< Maximum thrust [N]
  double max_torque = 12.0;            ///< Maximum torque per axis [N·m]

  // Physical constants
  double gravity = 9.81;               ///< Gravitational acceleration [m/s²]
  double quadcopter_mass = 1.0;        ///< Quadcopter mass (set via set_mass) [kg]
};

/// N-independent adaptive controller for cooperative lift.
///
/// This controller is designed to work without knowing N (number of quads)
/// or m_L (load mass). Instead, it relies on:
///   1. θ̂ estimate from AdaptiveLoadEstimator ≈ m_L/N
///   2. Local cable tension and direction measurements
///   3. Estimated load position and velocity (from DecentralizedLoadEstimator)
///
/// Control law structure:
///   F_quad = m_Q * (g + a_position) + θ̂ * g * n + F_tension_regulation
///
/// where:
///   - m_Q * (g + a_position): Quad's own dynamics compensation
///   - θ̂ * g * n: Adaptive load weight share along cable
///   - F_tension_regulation: Maintains desired cable tension
///
/// Input ports:
///   - quad_state: Quadcopter [px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz] (13)
///   - cable_tension: Rope tension magnitude [N] (1)
///   - cable_direction: Unit vector from quad toward load (3)
///   - theta_hat: Adaptive estimate of m_L/N [kg] (1)
///   - load_position_est: Estimated load position [m] (3)
///   - load_velocity_est: Estimated load velocity [m/s] (3)
///   - load_trajectory_des: Desired load position and velocity [m, m/s] (6)
///
/// Output ports:
///   - control_force: Thrust and torques applied to quadcopter (spatial force)
///
class AdaptiveLiftController final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AdaptiveLiftController);

  /// Constructs the adaptive lift controller.
  ///
  /// @param plant The MultibodyPlant containing the quadcopter.
  /// @param quadcopter_body The quadcopter RigidBody.
  /// @param params Controller parameters.
  AdaptiveLiftController(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::multibody::RigidBody<double>& quadcopter_body,
      const AdaptiveLiftControllerParams& params = AdaptiveLiftControllerParams());

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_quad_state_input_port() const {
    return get_input_port(quad_state_port_);
  }
  const drake::systems::InputPort<double>& get_cable_tension_input_port() const {
    return get_input_port(cable_tension_port_);
  }
  const drake::systems::InputPort<double>& get_cable_direction_input_port() const {
    return get_input_port(cable_direction_port_);
  }
  const drake::systems::InputPort<double>& get_theta_hat_input_port() const {
    return get_input_port(theta_hat_port_);
  }
  const drake::systems::InputPort<double>& get_load_position_est_input_port() const {
    return get_input_port(load_position_est_port_);
  }
  const drake::systems::InputPort<double>& get_load_velocity_est_input_port() const {
    return get_input_port(load_velocity_est_port_);
  }
  const drake::systems::InputPort<double>& get_load_trajectory_des_input_port() const {
    return get_input_port(load_trajectory_des_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_control_output_port() const {
    return get_output_port(control_port_);
  }

  /// Update the quadcopter mass (call after welding visual model).
  void set_mass(double mass);

  /// Get current parameters.
  const AdaptiveLiftControllerParams& params() const { return params_; }

 private:
  // Output computation
  void CalcControlForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  // Compute desired load trajectory at time t (if not using external trajectory)
  void ComputeLoadTrajectory(double t,
                             Eigen::Vector3d& pos_des,
                             Eigen::Vector3d& vel_des) const;

  // Extract quadcopter state from input
  void ExtractQuadState(const Eigen::VectorXd& state,
                        Eigen::Vector3d& position,
                        Eigen::Quaterniond& orientation,
                        Eigen::Vector3d& velocity,
                        Eigen::Vector3d& angular_velocity) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Quadcopter body index
  drake::multibody::BodyIndex quad_body_index_;

  // Parameters
  AdaptiveLiftControllerParams params_;

  // Port indices
  drake::systems::InputPortIndex quad_state_port_;
  drake::systems::InputPortIndex cable_tension_port_;
  drake::systems::InputPortIndex cable_direction_port_;
  drake::systems::InputPortIndex theta_hat_port_;
  drake::systems::InputPortIndex load_position_est_port_;
  drake::systems::InputPortIndex load_velocity_est_port_;
  drake::systems::InputPortIndex load_trajectory_des_port_;

  drake::systems::OutputPortIndex control_port_;

  // Gravity
  static constexpr double kGravity = 9.81;
};

}  // namespace quad_rope_lift
