#pragma once

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Dense>

namespace tether_lift {

/**
 * @brief Computes the required total cable force for load trajectory tracking.
 *
 * Implements Equation (5) from the mathematical framework:
 *   F_req = m̂_L * (ä_L^d + g·e₃ - Kp·e_L - Kd·ė_L)
 *
 * where:
 *   - m̂_L = N·θ̂ is the estimated load mass (θ̂ estimated per-drone)
 *   - ä_L^d is the desired load acceleration (feedforward)
 *   - g·e₃ is gravity compensation
 *   - Kp, Kd are positive definite feedback gains
 *   - e_L = p_L - p_L^d is position error
 *   - ė_L = v_L - v_L^d is velocity error
 *
 * Input ports:
 *   0: load_trajectory (9D: p_L^d, v_L^d, ä_L^d)
 *   1: load_state (6D: p_L, v_L - actual load position/velocity)
 *   2: theta_hat (scalar: adaptive parameter θ̂ = m_L/N)
 *   3: num_drones (scalar: N)
 *
 * Output ports:
 *   0: required_force (3D: F_req vector)
 *   1: estimated_load_mass (scalar: m̂_L for downstream use)
 */
class RequiredForceComputer final : public drake::systems::LeafSystem<double> {
 public:
  struct Params {
    // PD feedback gains (diagonal elements for 3x3 matrices)
    Eigen::Vector3d Kp;
    Eigen::Vector3d Kd;

    // Gravity constant
    double gravity;

    // Minimum estimated mass (safety floor)
    double min_mass_estimate;

    // Maximum control force magnitude (saturation)
    double max_force_magnitude;

    Params()
        : Kp(3.0, 3.0, 5.0)      // Reduced for bandwidth separation (was 10,10,15)
        , Kd(5.0, 5.0, 6.0)      // Increased for higher damping ratio ζ≈0.8 (was 2,2,3)
        , gravity(9.81)
        , min_mass_estimate(0.1)
        , max_force_magnitude(500.0) {}
  };

  explicit RequiredForceComputer(const Params& params = Params());

  // Input port getters
  const drake::systems::InputPort<double>& get_load_trajectory_input() const {
    return get_input_port(load_trajectory_port_);
  }

  const drake::systems::InputPort<double>& get_load_state_input() const {
    return get_input_port(load_state_port_);
  }

  const drake::systems::InputPort<double>& get_theta_hat_input() const {
    return get_input_port(theta_hat_port_);
  }

  const drake::systems::InputPort<double>& get_num_drones_input() const {
    return get_input_port(num_drones_port_);
  }

  // Output port getters
  const drake::systems::OutputPort<double>& get_required_force_output() const {
    return get_output_port(required_force_port_);
  }

  const drake::systems::OutputPort<double>& get_estimated_mass_output() const {
    return get_output_port(estimated_mass_port_);
  }

  // Parameter accessors
  const Params& params() const { return params_; }
  void set_params(const Params& params) { params_ = params; }

 private:
  void CalcRequiredForce(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcEstimatedMass(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Helper to compute the estimated load mass from theta_hat and N
  double ComputeEstimatedMass(
      const drake::systems::Context<double>& context) const;

  Params params_;

  // Port indices
  int load_trajectory_port_{};
  int load_state_port_{};
  int theta_hat_port_{};
  int num_drones_port_{};
  int required_force_port_{};
  int estimated_mass_port_{};
};

}  // namespace tether_lift
