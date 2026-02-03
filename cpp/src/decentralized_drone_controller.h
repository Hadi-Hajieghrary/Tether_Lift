#pragma once

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Dense>

namespace tether_lift {

/**
 * @brief Fully decentralized drone controller for cooperative load transport.
 *
 * This controller requires NO knowledge of:
 *   - N (number of drones)
 *   - m_L (total load mass)
 *   - Other drones' states or actions
 *
 * Each drone uses ONLY:
 *   - Its own cable tension T_i (measured)
 *   - Its own cable direction q_i (measured)
 *   - Its own state (position, velocity)
 *   - Its attachment point on load a_i^L (fixed geometry)
 *   - Its cable length ℓ_i (fixed)
 *   - Adaptive parameter θ̂_i (learned, converges to m_L/N)
 *   - Load position/velocity (shared broadcast from load GPS)
 *   - Load trajectory (shared broadcast - same for all drones)
 *
 * Control Law:
 *   1. Compute load tracking error:
 *        e_L = p̂_L - p_L^d
 *        ė_L = v̂_L - v_L^d
 *
 *   2. Compute desired force THIS drone should apply:
 *        F_i^d = θ̂_i × (ä_L^d + g·e₃ - Kp·e_L - Kd·ė_L)
 *
 *   3. Extract desired tension and cable direction:
 *        T_i^d = ||F_i^d||
 *        q_i^d = F_i^d / T_i^d
 *
 *   4. Compute desired drone position:
 *        p_i^d = p̂_L + a_i^L + ℓ_i·q_i^d
 *
 *   5. Drone position tracking with cable compensation:
 *        f_i = m_i(a_i^d + g·e₃) + T_i·q_i - Kp_drone·e_i - Kd_drone·ė_i
 *
 * Key insight: When N identical drones run this policy with θ̂_i → m_L/N,
 * the total force automatically sums to the correct value:
 *   Σᵢ F_i = Σᵢ θ̂_i × (control) = m_L × (control) ✓
 *
 * Input ports:
 *   0: drone_state (13D: position[3], quaternion[4], velocity[3], angular_velocity[3])
 *   1: load_state (6D: p_L[3], v_L[3] - from shared GPS broadcast)
 *   2: load_trajectory (9D: p_L^d[3], v_L^d[3], a_L^d[3] - shared broadcast)
 *   3: theta_hat (1D: adaptive parameter θ̂_i - from AdaptiveLoadEstimator)
 *   4: cable_tension (1D: measured T_i)
 *   5: cable_direction (3D: measured q_i, pointing from load to drone)
 *
 * Output ports:
 *   0: thrust_force (3D: desired thrust vector in world frame)
 *   1: desired_drone_position (3D: p_i^d for logging/debugging)
 *   2: desired_cable_direction (3D: q_i^d for logging/debugging)
 *   3: desired_tension (1D: T_i^d for logging/debugging)
 */
class DecentralizedDroneController final : public drake::systems::LeafSystem<double> {
 public:
  struct Params {
    // === Drone physical properties ===
    double drone_mass;           // kg

    // === Cable geometry (known to this drone only) ===
    Eigen::Vector3d attachment_point;  // a_i^L in load frame
    double cable_length;               // ℓ_i in meters

    // === Load tracking gains (shared, but no communication needed) ===
    Eigen::Vector3d Kp_load;     // Position error gain on load
    Eigen::Vector3d Kd_load;     // Velocity error gain on load

    // === Drone position tracking gains ===
    Eigen::Vector3d Kp_drone;    // Position error gain on drone
    Eigen::Vector3d Kd_drone;    // Velocity error gain on drone

    // === Physical constants ===
    double gravity;

    // === Safety limits ===
    double max_thrust;           // Maximum thrust magnitude [N]
    double min_thrust;           // Minimum thrust [N]
    double min_tension;          // Minimum cable tension [N]

    // === Cable compensation ===
    double cable_compensation_gain;  // 0 to 1

    Params()
        : drone_mass(1.0)
        , attachment_point(0.0, 0.0, 0.0)
        , cable_length(3.0)
        , Kp_load(8.0, 8.0, 12.0)
        , Kd_load(4.0, 4.0, 6.0)
        , Kp_drone(15.0, 15.0, 20.0)
        , Kd_drone(8.0, 8.0, 10.0)
        , gravity(9.81)
        , max_thrust(30.0)
        , min_thrust(0.1)
        , min_tension(0.5)
        , cable_compensation_gain(1.0) {}
  };

  explicit DecentralizedDroneController(const Params& params = Params());

  // === Input port getters ===
  const drake::systems::InputPort<double>& get_drone_state_input() const {
    return get_input_port(drone_state_port_);
  }

  const drake::systems::InputPort<double>& get_load_state_input() const {
    return get_input_port(load_state_port_);
  }

  const drake::systems::InputPort<double>& get_load_trajectory_input() const {
    return get_input_port(load_trajectory_port_);
  }

  const drake::systems::InputPort<double>& get_theta_hat_input() const {
    return get_input_port(theta_hat_port_);
  }

  const drake::systems::InputPort<double>& get_cable_tension_input() const {
    return get_input_port(cable_tension_port_);
  }

  const drake::systems::InputPort<double>& get_cable_direction_input() const {
    return get_input_port(cable_direction_port_);
  }

  // === Output port getters ===
  const drake::systems::OutputPort<double>& get_thrust_force_output() const {
    return get_output_port(thrust_force_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_position_output() const {
    return get_output_port(desired_position_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_direction_output() const {
    return get_output_port(desired_direction_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_tension_output() const {
    return get_output_port(desired_tension_port_);
  }

  const Params& params() const { return params_; }

 private:
  /**
   * @brief Core computation: from load error to desired cable force.
   *
   * F_i^d = θ̂_i × (ä_L^d + g·e₃ - Kp·e_L - Kd·ė_L)
   */
  Eigen::Vector3d ComputeDesiredCableForce(
      const drake::systems::Context<double>& context) const;

  /**
   * @brief Compute desired drone position from load state and cable geometry.
   *
   * p_i^d = p̂_L + a_i^L + ℓ_i·q_i^d
   */
  Eigen::Vector3d ComputeDesiredDronePosition(
      const Eigen::Vector3d& load_position,
      const Eigen::Vector3d& desired_cable_direction) const;

  void CalcThrustForce(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredPosition(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredDirection(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredTension(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  Params params_;

  // Port indices
  int drone_state_port_{};
  int load_state_port_{};
  int load_trajectory_port_{};
  int theta_hat_port_{};
  int cable_tension_port_{};
  int cable_direction_port_{};
  int thrust_force_port_{};
  int desired_position_port_{};
  int desired_direction_port_{};
  int desired_tension_port_{};
};

}  // namespace tether_lift
