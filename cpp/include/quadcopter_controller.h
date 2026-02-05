#pragma once

#include <Eigen/Core>
#include <vector>
#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// A waypoint in the trajectory with position and timing.
struct TrajectoryWaypoint {
  Eigen::Vector3d position{0, 0, 1};  ///< Target position [x, y, z] in world frame [m]
  double arrival_time = 0.0;           ///< Time to arrive at this waypoint [s]
  double hold_time = 0.0;              ///< Time to hold at this waypoint before moving to next [s]
};

/// Controller parameters struct for cleaner initialization.
struct ControllerParams {
  // Formation offset (relative to trajectory waypoints)
  Eigen::Vector3d formation_offset{0, 0, 0};  ///< Offset from shared trajectory [m]

  // Trajectory parameters (legacy - used if waypoints empty)
  double initial_altitude = 1.0;       ///< Starting hover height [m]
  double final_altitude = 3.0;         ///< Target altitude after lifting payload [m]
  double ascent_start_time = 0.5;      ///< Time to begin ascending [s]
  double climb_rate = 0.4;             ///< Vertical velocity during ascent [m/s]

  // Waypoint-based trajectory (if non-empty, overrides legacy trajectory)
  std::vector<TrajectoryWaypoint> waypoints;

  // Position control gains (x/y)
  // TUNED: Aggressive gains for faster horizontal tracking
  double position_kp = 15.0;           ///< Proportional gain for x/y position control
  double position_kd = 7.0;            ///< Derivative gain for x/y position control
  double max_tilt_angle = 0.4;         ///< Maximum tilt angle for x/y control [rad] (~23 deg)

  // Altitude control gains (z)
  // TUNED: High gains for rapid vertical response
  double altitude_kp = 25.0;           ///< Proportional gain for altitude control
  double altitude_kd = 12.0;           ///< Derivative gain for altitude control

  // Attitude control gains
  // TUNED: High gains for snappy attitude response
  double attitude_kp = 15.0;           ///< Proportional gain for attitude control
  double attitude_kd = 2.5;            ///< Derivative gain for attitude control

  // Tension feedback gains
  // TUNED: Aggressive for quick load engagement
  double tension_feedback_kp = 1.5;    ///< Gain for tension error feedback to thrust
  double tension_altitude_gain = 0.008;///< Gain for tension error to altitude adjustment
  double tension_altitude_max = 1.0;   ///< Max altitude adjustment from tension [m]

  // Pickup phase parameters
  // TUNED: Fast pickup ramp
  double pickup_ramp_duration = 1.0;   ///< Time to ramp up tension target during pickup [s]
  double pickup_target_tension = 20.0; ///< Final target rope tension (≈ payload weight/N_quads) [N]
  double pickup_detection_threshold = 0.3; ///< Tension level that triggers pickup mode [N]

  // Actuator limits
  double min_thrust = 0.0;             ///< Minimum allowed thrust [N]
  double max_thrust = 150.0;           ///< Maximum allowed thrust [N]
  double max_torque = 10.0;            ///< Maximum allowed torque per axis [N·m]

  // Physical constants
  double gravity = 9.81;               ///< Gravitational acceleration [m/s²]
};

/// Controller for quadcopter payload pickup and lift.
///
/// This controller implements:
/// 1. Position control: PD controller to track x/y/z trajectory with formation offset.
/// 2. Attitude control: PD controller to maintain desired orientation.
/// 3. Tension-aware pickup: Smooth load transfer using rope tension feedback.
///
/// Supports both legacy single-altitude trajectory and waypoint-based trajectories.
class QuadcopterLiftController final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuadcopterLiftController);

  /// Constructs the quadcopter controller.
  ///
  /// @param plant The MultibodyPlant containing the quadcopter.
  /// @param quadcopter_body The quadcopter RigidBody.
  /// @param params Controller parameters.
  QuadcopterLiftController(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::multibody::RigidBody<double>& quadcopter_body,
      const ControllerParams& params = ControllerParams());

  /// Returns the input port for plant state.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Returns the input port for rope tension [tension, fx, fy, fz].
  const drake::systems::InputPort<double>& get_tension_input_port() const {
    return get_input_port(tension_port_);
  }

  /// Returns the input port for estimated state [px, py, pz, vx, vy, vz].
  /// When connected, the controller uses this instead of extracting from plant_state.
  const drake::systems::InputPort<double>& get_estimated_state_input_port() const {
    return get_input_port(estimated_state_port_);
  }

  /// Returns the output port for control forces.
  const drake::systems::OutputPort<double>& get_control_output_port() const {
    return get_output_port(control_port_);
  }

  /// Returns the output port for control vector [tau_x, tau_y, tau_z, f_x, f_y, f_z].
  /// This is for logging purposes.
  const drake::systems::OutputPort<double>& get_control_vector_output_port() const {
    return get_output_port(control_vector_port_);
  }

  /// Update the quadcopter mass (call after welding visual model).
  void set_mass(double mass) { mass_ = mass; }

 private:
  // Output computation
  void CalcControlForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  void CalcControlVector(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  /// Compute desired position and velocity at time t.
  void ComputeTrajectory(double t, Eigen::Vector3d& pos_des, Eigen::Vector3d& vel_des) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Quadcopter body index
  drake::multibody::BodyIndex quad_body_index_;

  // Mass (mutable for set_mass)
  mutable double mass_;

  // Formation offset
  Eigen::Vector3d formation_offset_;

  // Waypoint trajectory
  std::vector<TrajectoryWaypoint> waypoints_;
  bool use_waypoints_;

  // Legacy trajectory parameters
  double initial_altitude_;
  double final_altitude_;
  double ascent_start_time_;
  double climb_rate_;
  double ascent_duration_;
  double ascent_direction_;

  // Control gains
  double position_kp_;
  double position_kd_;
  double max_tilt_angle_;
  double altitude_kp_;
  double altitude_kd_;
  double attitude_kp_;
  double attitude_kd_;
  double tension_kp_;
  double tension_altitude_gain_;
  double tension_altitude_max_;

  // Pickup phase parameters
  double pickup_duration_;
  double pickup_target_tension_;
  double pickup_threshold_;

  // Actuator limits
  double min_thrust_;
  double max_thrust_;
  double max_torque_;

  // Physical constants
  double gravity_;

  // Mutable state for pickup detection
  mutable std::optional<double> pickup_start_time_;

  // Port indices
  int plant_state_port_{-1};
  int tension_port_{-1};
  int estimated_state_port_{-1};
  int control_port_{-1};
  int control_vector_port_{-1};
};

}  // namespace quad_rope_lift
