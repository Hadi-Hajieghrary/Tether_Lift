#pragma once

/// @file cbf_safety_filter.h
/// @brief Control Barrier Function (CBF) safety filter for cable constraints.
///
/// This safety filter modifies nominal control inputs to ensure:
///   1. Cable tension stays within bounds (no slack, no excessive tension)
///   2. Cable angles stay within safe limits
///   3. Inter-quadcopter spacing is maintained
///
/// Uses Control Barrier Functions (Ames et al., 2017) to provide
/// provable safety guarantees with minimal intervention on nominal control.

#include <Eigen/Core>
#include <vector>

#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

/// Parameters for the CBF safety filter.
struct CbfSafetyParams {
  // Cable tension constraints
  double min_tension = 1.0;           ///< Minimum cable tension [N]
  double max_tension = 50.0;          ///< Maximum cable tension [N]
  double tension_rate_limit = 30.0;   ///< Maximum tension change rate [N/s]

  // Cable angle constraints
  double max_cable_angle = 0.7;       ///< Max cable angle from vertical [rad] (~40°)
  double cable_angle_margin = 0.1;    ///< Safety margin on angle constraint [rad]

  // Inter-vehicle constraints
  double min_quad_spacing = 0.8;      ///< Minimum distance between quads [m]
  double collision_barrier_margin = 0.3; ///< Safety margin for collision [m]

  // CBF parameters
  double cbf_alpha = 5.0;             ///< CBF class-K function gain (higher = more aggressive)
  double cbf_relaxation_weight = 100.0;///< Weight on slack variable in QP
  double nominal_weight = 1.0;         ///< Weight on tracking nominal control

  // QP solver parameters
  double qp_tolerance = 1e-6;         ///< Solver tolerance
  int qp_max_iterations = 100;        ///< Maximum QP iterations

  // Constraint activation thresholds
  double tension_activation_margin = 5.0; ///< Start filtering N before limit
  double angle_activation_margin = 0.15;  ///< Start filtering rad before limit
};

/// Control Barrier Function safety filter.
///
/// This system filters nominal control commands to maintain safety.
/// It solves a QP at each timestep:
///
///   min  ||u - u_nom||² + λ||δ||²
///   s.t. Lf h(x) + Lg h(x) u + α(h(x)) ≥ -δ  (CBF constraint)
///        u_min ≤ u ≤ u_max                    (actuator limits)
///
/// Where h(x) > 0 defines the safe set.
///
/// Barrier functions implemented:
///   1. Tension lower bound: h_Tmin = T - T_min
///   2. Tension upper bound: h_Tmax = T_max - T
///   3. Cable angle: h_angle = cos(θ_max) - cos(θ)
///   4. Collision avoidance: h_coll = ||r_i - r_j||² - d_min²
///
/// Input ports:
///   - nominal_thrust: Nominal thrust command [N] (1)
///   - nominal_torque: Nominal torque commands [N·m] (3)
///   - quad_state: Quadcopter state [p, q, v, ω] (13)
///   - cable_tension: Current cable tension [N] (1)
///   - cable_direction: Unit vector from quad to load (3)
///   - neighbor_positions: Positions of nearby quadcopters (3*N_neighbors)
///
/// Output ports:
///   - filtered_thrust: Safe thrust command [N] (1)
///   - filtered_torque: Safe torque commands [N·m] (3)
///   - constraint_active: Which constraints are active (4)
///   - barrier_values: Current barrier function values (4)
///
class CbfSafetyFilter final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CbfSafetyFilter);

  /// Constructs the CBF safety filter.
  ///
  /// @param quadcopter_mass Mass of the quadcopter [kg]
  /// @param max_neighbors Maximum number of neighbors to consider
  /// @param params Safety filter parameters
  CbfSafetyFilter(double quadcopter_mass,
                  int max_neighbors = 3,
                  const CbfSafetyParams& params = CbfSafetyParams());

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_nominal_thrust_input_port() const {
    return get_input_port(nominal_thrust_port_);
  }
  const drake::systems::InputPort<double>& get_nominal_torque_input_port() const {
    return get_input_port(nominal_torque_port_);
  }
  const drake::systems::InputPort<double>& get_quad_state_input_port() const {
    return get_input_port(quad_state_port_);
  }
  const drake::systems::InputPort<double>& get_cable_tension_input_port() const {
    return get_input_port(cable_tension_port_);
  }
  const drake::systems::InputPort<double>& get_cable_direction_input_port() const {
    return get_input_port(cable_direction_port_);
  }
  const drake::systems::InputPort<double>& get_neighbor_positions_input_port() const {
    return get_input_port(neighbor_positions_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_filtered_thrust_output_port() const {
    return get_output_port(filtered_thrust_port_);
  }
  const drake::systems::OutputPort<double>& get_filtered_torque_output_port() const {
    return get_output_port(filtered_torque_port_);
  }
  const drake::systems::OutputPort<double>& get_constraint_active_output_port() const {
    return get_output_port(constraint_active_port_);
  }
  const drake::systems::OutputPort<double>& get_barrier_values_output_port() const {
    return get_output_port(barrier_values_port_);
  }

  /// Get parameters.
  const CbfSafetyParams& params() const { return params_; }

  /// Update quadcopter mass.
  void set_mass(double mass) { quadcopter_mass_ = mass; }

 private:
  // Output calculations
  void CalcFilteredThrust(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;
  void CalcFilteredTorque(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;
  void CalcConstraintActive(const drake::systems::Context<double>& context,
                            drake::systems::BasicVector<double>* output) const;
  void CalcBarrierValues(const drake::systems::Context<double>& context,
                         drake::systems::BasicVector<double>* output) const;

  // Barrier function calculations
  double ComputeTensionLowerBarrier(double T) const;
  double ComputeTensionUpperBarrier(double T) const;
  double ComputeCableAngleBarrier(const Eigen::Vector3d& cable_dir) const;
  double ComputeCollisionBarrier(const Eigen::Vector3d& my_pos,
                                 const Eigen::Vector3d& neighbor_pos) const;

  // QP solver (simplified - active set method for small QP)
  struct QpSolution {
    double filtered_thrust;
    Eigen::Vector3d filtered_torque;
    Eigen::Vector4d constraint_active;  // Which constraints are binding
    bool success;
  };

  QpSolution SolveQp(
      double nominal_thrust,
      const Eigen::Vector3d& nominal_torque,
      const Eigen::Vector3d& quad_pos,
      const Eigen::Vector3d& quad_vel,
      double cable_tension,
      const Eigen::Vector3d& cable_dir,
      const std::vector<Eigen::Vector3d>& neighbor_positions) const;

  // Parameters
  CbfSafetyParams params_;
  double quadcopter_mass_;
  int max_neighbors_;

  // Actuator limits
  double min_thrust_ = 0.0;
  double max_thrust_ = 150.0;
  double max_torque_ = 10.0;

  // Port indices
  drake::systems::InputPortIndex nominal_thrust_port_;
  drake::systems::InputPortIndex nominal_torque_port_;
  drake::systems::InputPortIndex quad_state_port_;
  drake::systems::InputPortIndex cable_tension_port_;
  drake::systems::InputPortIndex cable_direction_port_;
  drake::systems::InputPortIndex neighbor_positions_port_;

  drake::systems::OutputPortIndex filtered_thrust_port_;
  drake::systems::OutputPortIndex filtered_torque_port_;
  drake::systems::OutputPortIndex constraint_active_port_;
  drake::systems::OutputPortIndex barrier_values_port_;

  // Number of barrier constraints
  static constexpr int kNumBarriers = 4;  // T_min, T_max, angle, collision
};

}  // namespace quad_rope_lift
