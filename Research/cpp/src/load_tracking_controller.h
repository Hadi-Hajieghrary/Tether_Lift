#pragma once

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Dense>

namespace tether_lift {

/**
 * @brief Full load trajectory tracking controller for individual quadcopter.
 *
 * Implements Equation (21) from the mathematical framework:
 *
 *   f_i^d = m_i·(a_i^d + g·e₃) + T_i^d·q_i^d - Kp_i·e_i - Kd_i·ė_i
 *
 * where:
 *   - m_i is the drone mass
 *   - a_i^d is the desired drone acceleration (from DroneTrajectoryMapper)
 *   - g·e₃ is gravity compensation
 *   - T_i^d·q_i^d is the desired cable tension force compensation
 *   - e_i = p_i - p_i^d is position error
 *   - ė_i = v_i - v_i^d is velocity error
 *   - Kp_i, Kd_i are feedback gains
 *
 * The controller also computes the required attitude (thrust direction)
 * for the quadcopter's geometric controller.
 *
 * Key insight from theory: By tracking the computed p_i^d (which depends on
 * load trajectory AND cable geometry), each drone automatically applies the
 * correct force to track the load trajectory.
 *
 * Input ports:
 *   0: drone_state (13D: position[3], quaternion[4], velocity[3], angular_velocity[3])
 *   1: drone_trajectory (9D: p_i^d, v_i^d, a_i^d from DroneTrajectoryMapper)
 *   2: desired_tension (1D: T_i^d from ForceAllocationSystem)
 *   3: desired_cable_direction (3D: q_i^d from ForceAllocationSystem)
 *   4: actual_cable_direction (3D: q_i actual cable direction for compensation)
 *   5: actual_tension (1D: T_i actual cable tension for feedforward)
 *
 * Output ports:
 *   0: thrust_force (3D: f_i^d desired thrust vector in world frame)
 *   1: thrust_magnitude (1D: ||f_i^d||)
 *   2: desired_attitude (4D: quaternion representing desired thrust direction)
 *   3: desired_angular_velocity (3D: for attitude controller)
 */
class LoadTrackingController final : public drake::systems::LeafSystem<double> {
 public:
  struct Params {
    double mass;               // kg
    Eigen::Vector3d Kp;        // Position feedback gains (diagonal)
    Eigen::Vector3d Kd;        // Velocity feedback gains (diagonal)
    double gravity;
    bool use_actual_tension_feedforward;  // Use actual vs desired tension
    double cable_compensation_gain;       // 0 = no compensation, 1 = full
    double max_thrust;         // N (safety limit)
    double min_thrust;         // N (ensure positive thrust)
    Eigen::Vector3d Ki;        // Integral gain (optional)
    double max_integral;

    Params()
        : mass(1.0)
        , Kp(25.0, 25.0, 35.0)   // Increased for faster inner loop (was 15,15,20)
        , Kd(12.0, 12.0, 15.0)   // Increased for better damping (was 8,8,10)
        , gravity(9.81)
        , use_actual_tension_feedforward(true)
        , cable_compensation_gain(1.0)
        , max_thrust(50.0)       // Increased for higher gains (was 30)
        , min_thrust(0.1)
        , Ki(0.0, 0.0, 0.0)
        , max_integral(5.0) {}
  };

  explicit LoadTrackingController(const Params& params = Params());

  // Input port getters
  const drake::systems::InputPort<double>& get_drone_state_input() const {
    return get_input_port(drone_state_port_);
  }

  const drake::systems::InputPort<double>& get_drone_trajectory_input() const {
    return get_input_port(drone_trajectory_port_);
  }

  const drake::systems::InputPort<double>& get_desired_tension_input() const {
    return get_input_port(desired_tension_port_);
  }

  const drake::systems::InputPort<double>& get_desired_cable_direction_input() const {
    return get_input_port(desired_cable_direction_port_);
  }

  const drake::systems::InputPort<double>& get_actual_cable_direction_input() const {
    return get_input_port(actual_cable_direction_port_);
  }

  const drake::systems::InputPort<double>& get_actual_tension_input() const {
    return get_input_port(actual_tension_port_);
  }

  // Output port getters
  const drake::systems::OutputPort<double>& get_thrust_force_output() const {
    return get_output_port(thrust_force_port_);
  }

  const drake::systems::OutputPort<double>& get_thrust_magnitude_output() const {
    return get_output_port(thrust_magnitude_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_attitude_output() const {
    return get_output_port(desired_attitude_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_angular_velocity_output() const {
    return get_output_port(desired_angular_velocity_port_);
  }

  // Parameter accessors
  const Params& params() const { return params_; }
  void set_params(const Params& params) { params_ = params; }

 private:
  // Continuous state for integral term
  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  void DoCalcTimeDerivatives(
      const drake::systems::Context<double>& context,
      drake::systems::ContinuousState<double>* derivatives) const override;

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

  // Core control computation
  Eigen::Vector3d ComputeThrustVector(
      const drake::systems::Context<double>& context) const;

  // Convert thrust vector to quaternion (thrust direction)
  Eigen::Quaterniond ThrustVectorToQuaternion(
      const Eigen::Vector3d& thrust,
      double yaw_des = 0.0) const;

  Params params_;

  // Port indices
  int drone_state_port_{};
  int drone_trajectory_port_{};
  int desired_tension_port_{};
  int desired_cable_direction_port_{};
  int actual_cable_direction_port_{};
  int actual_tension_port_{};
  int thrust_force_port_{};
  int thrust_magnitude_port_{};
  int desired_attitude_port_{};
  int desired_angular_velocity_port_{};
};

}  // namespace tether_lift
