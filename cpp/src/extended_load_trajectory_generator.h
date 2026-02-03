#pragma once

#include "load_trajectory_generator.h"
#include <Eigen/Dense>
#include <functional>

namespace tether_lift {

/**
 * @brief Extended load trajectory generator with jerk output and feasibility checking.
 *
 * Extends quad_rope_lift::LoadTrajectoryGenerator with:
 * - Jerk output for cable direction derivative computation (q̇_i^d)
 * - Tension feasibility checking (Equation 9)
 * - Horizontal acceleration limit validation: ||ä_L_horiz|| < g·tan(α_max)
 * - Full trajectory output in single port (12D: position, velocity, acceleration, jerk)
 *
 * This system wraps the base generator and adds the additional outputs
 * required for load trajectory following per the mathematical framework.
 *
 * Output ports:
 *   0: full_trajectory (12D: [p_L^d, v_L^d, a_L^d, j_L^d])
 *   1: feasibility (1D: 1.0 if feasible, 0.0 if not)
 *   2: max_cable_angle (1D: maximum cable angle from vertical [rad])
 */
class ExtendedLoadTrajectoryGenerator final : public drake::systems::LeafSystem<double> {
 public:
  struct FeasibilityParams {
    double gravity;
    double max_cable_angle;       // Maximum angle from vertical [rad]
    double min_tension_ratio;     // Minimum T / (m_L * g / N)
    double max_tension_ratio;     // Maximum T / (m_L * g / N)
    double estimated_load_mass;   // kg (for feasibility check only)
    int num_drones;               // Number of drones

    FeasibilityParams()
        : gravity(9.81)
        , max_cable_angle(0.785398)  // 45 deg
        , min_tension_ratio(0.1)
        , max_tension_ratio(3.0)
        , estimated_load_mass(2.0)
        , num_drones(4) {}
  };

  /**
   * @brief Construct from waypoints.
   */
  ExtendedLoadTrajectoryGenerator(
      const std::vector<quad_rope_lift::LoadWaypoint>& waypoints,
      const quad_rope_lift::LoadTrajectoryParams& traj_params = quad_rope_lift::LoadTrajectoryParams(),
      const FeasibilityParams& feas_params = FeasibilityParams());

  /**
   * @brief Factory for altitude ramp trajectory.
   */
  static ExtendedLoadTrajectoryGenerator CreateAltitudeRamp(
      const Eigen::Vector3d& initial_pos,
      const Eigen::Vector3d& final_pos,
      double start_time,
      const quad_rope_lift::LoadTrajectoryParams& traj_params = quad_rope_lift::LoadTrajectoryParams(),
      const FeasibilityParams& feas_params = FeasibilityParams());

  /**
   * @brief Factory for hover trajectory.
   */
  static ExtendedLoadTrajectoryGenerator CreateHover(
      const Eigen::Vector3d& hover_position,
      const FeasibilityParams& feas_params = FeasibilityParams());

  // Output port getters
  const drake::systems::OutputPort<double>& get_full_trajectory_output() const {
    return get_output_port(full_trajectory_port_);
  }

  const drake::systems::OutputPort<double>& get_feasibility_output() const {
    return get_output_port(feasibility_port_);
  }

  const drake::systems::OutputPort<double>& get_max_angle_output() const {
    return get_output_port(max_angle_port_);
  }

  // Compatibility outputs (same as base generator)
  const drake::systems::OutputPort<double>& get_position_output() const {
    return get_output_port(position_port_);
  }

  const drake::systems::OutputPort<double>& get_velocity_output() const {
    return get_output_port(velocity_port_);
  }

  const drake::systems::OutputPort<double>& get_acceleration_output() const {
    return get_output_port(acceleration_port_);
  }

  const drake::systems::OutputPort<double>& get_jerk_output() const {
    return get_output_port(jerk_port_);
  }

  // Parameter accessors
  const FeasibilityParams& feasibility_params() const { return feas_params_; }

 private:
  void CalcFullTrajectory(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcFeasibility(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcMaxAngle(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcPosition(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcVelocity(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcAcceleration(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcJerk(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Evaluate full trajectory at time t
  void EvaluateTrajectoryWithJerk(
      double t,
      Eigen::Vector3d& position,
      Eigen::Vector3d& velocity,
      Eigen::Vector3d& acceleration,
      Eigen::Vector3d& jerk) const;

  // Check feasibility of acceleration
  // Returns maximum cable angle required (should be < max_cable_angle)
  double ComputeRequiredCableAngle(const Eigen::Vector3d& acceleration) const;

  // Base generator (composition)
  std::unique_ptr<quad_rope_lift::LoadTrajectoryGenerator> base_generator_;

  // Parameters
  FeasibilityParams feas_params_;

  // Small time delta for numerical differentiation of jerk
  static constexpr double kJerkDeltaT = 0.001;

  // Port indices
  int full_trajectory_port_{};
  int feasibility_port_{};
  int max_angle_port_{};
  int position_port_{};
  int velocity_port_{};
  int acceleration_port_{};
  int jerk_port_{};
};

}  // namespace tether_lift
