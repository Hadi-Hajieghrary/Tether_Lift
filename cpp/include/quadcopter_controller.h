#pragma once

#include <Eigen/Core>
#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Controller parameters struct for cleaner initialization.
struct ControllerParams {
  // Trajectory parameters
  double initial_altitude = 1.0;       ///< Starting hover height [m]
  double final_altitude = 3.0;         ///< Target altitude after lifting payload [m]
  double ascent_start_time = 0.5;      ///< Time to begin ascending [s]
  double climb_rate = 0.4;             ///< Vertical velocity during ascent [m/s]

  // Altitude control gains
  double altitude_kp = 15.0;           ///< Proportional gain for altitude control
  double altitude_kd = 8.0;            ///< Derivative gain for altitude control

  // Attitude control gains
  double attitude_kp = 8.0;            ///< Proportional gain for attitude control
  double attitude_kd = 1.5;            ///< Derivative gain for attitude control

  // Tension feedback gains
  double tension_feedback_kp = 0.5;    ///< Gain for tension error feedback to thrust
  double tension_altitude_gain = 0.003;///< Gain for tension error to altitude adjustment
  double tension_altitude_max = 0.5;   ///< Max altitude adjustment from tension [m]

  // Pickup phase parameters
  double pickup_ramp_duration = 2.0;   ///< Time to ramp up tension target during pickup [s]
  double pickup_target_tension = 20.0; ///< Final target rope tension (≈ payload weight) [N]
  double pickup_detection_threshold = 1.0; ///< Tension level that triggers pickup mode [N]

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
/// 1. Altitude control: PD controller to track a height trajectory.
/// 2. Attitude control: PD controller to maintain upright orientation.
/// 3. Tension-aware pickup: Smooth load transfer using rope tension feedback.
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

  /// Returns the output port for control forces.
  const drake::systems::OutputPort<double>& get_control_output_port() const {
    return get_output_port(control_port_);
  }

  /// Update the quadcopter mass (call after welding visual model).
  void set_mass(double mass) { mass_ = mass; }

 private:
  // Output computation
  void CalcControlForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Quadcopter body index
  drake::multibody::BodyIndex quad_body_index_;

  // Mass (mutable for set_mass)
  mutable double mass_;

  // Trajectory parameters
  double initial_altitude_;
  double final_altitude_;
  double ascent_start_time_;
  double climb_rate_;
  double ascent_duration_;
  double ascent_direction_;

  // Control gains
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
  int control_port_{-1};
};

}  // namespace quad_rope_lift
