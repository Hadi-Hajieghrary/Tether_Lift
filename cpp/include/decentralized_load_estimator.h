#pragma once

/// @file decentralized_load_estimator.h
/// @brief Decentralized load state estimation from local cable observations.
///
/// Each quadcopter estimates the load position/velocity without direct
/// communication with other agents. Uses cable direction and length
/// measurements combined with onboard position estimate.

#include <Eigen/Dense>
#include <drake/common/drake_copyable.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the decentralized load estimator.
struct DecentralizedLoadEstimatorParams {
  /// Process noise standard deviation for position [m]
  double position_process_noise = 0.05;

  /// Process noise standard deviation for velocity [m/s]
  double velocity_process_noise = 0.2;

  /// Measurement noise standard deviation for cable direction [rad]
  double cable_direction_noise = 0.02;

  /// Measurement noise standard deviation for cable length [m]
  double cable_length_noise = 0.01;

  /// Position-from-catenary additional measurement noise [m]
  /// (accounts for sag model uncertainty)
  double catenary_position_noise = 0.15;

  /// Maximum innovation for outlier rejection [standard deviations]
  double max_innovation_sigma = 3.0;

  /// Velocity estimate damping factor (0 = pure kinematic, 1 = highly damped)
  double velocity_damping = 0.05;

  /// Trust factor for cable-based position vs. dead reckoning (0-1)
  /// Higher = more trust in cable measurements
  double cable_trust_factor = 0.9;
};

/// Decentralized load state estimator.
///
/// Each quadcopter runs this estimator locally to estimate the load's
/// state using only information available at the drone:
///   - Own position (from ESKF)
///   - Cable tension (from measurement/model)
///   - Cable direction (unit vector from quad to attachment)
///   - Rope length (measured or nominal)
///
/// The load position is computed geometrically:
///   p_L = p_Q - L * n
/// where n is the unit vector from quad toward load.
///
/// Load velocity is estimated using a kinematic filter with
/// optional damping toward the estimated quasi-static velocity.
///
/// State: [p_L(3), v_L(3)] - 6 states per drone
///
/// Input ports:
///   - quad_position: Quadcopter position estimate [m] (3)
///   - quad_velocity: Quadcopter velocity estimate [m/s] (3)
///   - cable_direction: Unit vector from quad to load attachment (3)
///   - cable_length: Rope length [m] (1)
///   - cable_tension: Optional tension for confidence weighting [N] (1)
///
/// Output ports:
///   - load_position_estimate: Estimated load position [m] (3)
///   - load_velocity_estimate: Estimated load velocity [m/s] (3)
///   - estimation_covariance: Diagonal of P matrix (6)
///
class DecentralizedLoadEstimator final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DecentralizedLoadEstimator);

  /// Constructs the decentralized load estimator.
  ///
  /// @param dt Update period [s]
  /// @param params Estimator parameters
  DecentralizedLoadEstimator(
      double dt,
      const DecentralizedLoadEstimatorParams& params =
          DecentralizedLoadEstimatorParams());

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_quad_position_input_port() const {
    return get_input_port(quad_position_port_);
  }
  const drake::systems::InputPort<double>& get_quad_velocity_input_port() const {
    return get_input_port(quad_velocity_port_);
  }
  const drake::systems::InputPort<double>& get_cable_direction_input_port() const {
    return get_input_port(cable_direction_port_);
  }
  const drake::systems::InputPort<double>& get_cable_length_input_port() const {
    return get_input_port(cable_length_port_);
  }
  const drake::systems::InputPort<double>& get_cable_tension_input_port() const {
    return get_input_port(cable_tension_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_load_position_output_port() const {
    return get_output_port(load_position_port_);
  }
  const drake::systems::OutputPort<double>& get_load_velocity_output_port() const {
    return get_output_port(load_velocity_port_);
  }
  const drake::systems::OutputPort<double>& get_covariance_output_port() const {
    return get_output_port(covariance_port_);
  }

  /// Set initial state estimate.
  void SetInitialState(drake::systems::Context<double>* context,
                       const Eigen::Vector3d& position,
                       const Eigen::Vector3d& velocity) const;

  /// Get current state estimate.
  Eigen::Vector3d GetLoadPosition(
      const drake::systems::Context<double>& context) const;
  Eigen::Vector3d GetLoadVelocity(
      const drake::systems::Context<double>& context) const;

  const DecentralizedLoadEstimatorParams& params() const { return params_; }

 private:
  // Discrete update: predict then update with cable measurement
  drake::systems::EventStatus UpdateEstimate(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculations
  void CalcLoadPosition(const drake::systems::Context<double>& context,
                        drake::systems::BasicVector<double>* output) const;
  void CalcLoadVelocity(const drake::systems::Context<double>& context,
                        drake::systems::BasicVector<double>* output) const;
  void CalcCovariance(const drake::systems::Context<double>& context,
                      drake::systems::BasicVector<double>* output) const;

  // Parameters
  DecentralizedLoadEstimatorParams params_;
  double dt_;

  // Port indices
  drake::systems::InputPortIndex quad_position_port_;
  drake::systems::InputPortIndex quad_velocity_port_;
  drake::systems::InputPortIndex cable_direction_port_;
  drake::systems::InputPortIndex cable_length_port_;
  drake::systems::InputPortIndex cable_tension_port_;

  drake::systems::OutputPortIndex load_position_port_;
  drake::systems::OutputPortIndex load_velocity_port_;
  drake::systems::OutputPortIndex covariance_port_;

  // Discrete state index
  // State layout: [p_L(3), v_L(3), P_diag(6)] = 12 states
  drake::systems::DiscreteStateIndex state_index_;

  // State offsets
  static constexpr int kPositionOffset = 0;
  static constexpr int kVelocityOffset = 3;
  static constexpr int kCovarianceOffset = 6;
  static constexpr int kStateSize = 12;
};

}  // namespace quad_rope_lift
