#pragma once

#include <Eigen/Core>
#include <optional>

#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the state estimator.
struct EstimatorParams {
  // Process noise (how much we trust the model)
  Eigen::Vector3d position_process_noise{0.01, 0.01, 0.01};  ///< Position process noise [m]
  Eigen::Vector3d velocity_process_noise{0.1, 0.1, 0.1};     ///< Velocity process noise [m/s]

  // Measurement noise (how much we trust GPS)
  Eigen::Vector3d gps_measurement_noise{0.02, 0.02, 0.05};   ///< GPS measurement noise [m]

  // Initial covariance
  double initial_position_variance = 0.1;   ///< Initial position uncertainty [m²]
  double initial_velocity_variance = 0.5;   ///< Initial velocity uncertainty [(m/s)²]
};

/// Simple Extended Kalman Filter for position and velocity estimation.
///
/// State: [px, py, pz, vx, vy, vz] (6 states)
/// Process model: constant velocity with process noise
/// Measurement: GPS position [px, py, pz]
///
/// This is a basic estimator that can be extended with:
/// - IMU propagation (requires adding IMU input)
/// - Attitude estimation (requires quaternion state)
/// - Cable constraint pseudo-measurements
///
/// Input ports:
///   - gps_position: GPS measurement [x, y, z]
///   - gps_valid: Flag indicating if GPS is valid (1.0 = valid)
///
/// Output ports:
///   - estimated_state: [px, py, pz, vx, vy, vz]
///   - covariance_diagonal: Diagonal of P matrix (for monitoring)
///
class PositionVelocityEstimator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PositionVelocityEstimator);

  /// Constructs the state estimator.
  ///
  /// @param dt Discrete update period [s]
  /// @param params Estimator parameters
  PositionVelocityEstimator(double dt, const EstimatorParams& params = EstimatorParams());

  /// Returns the GPS position input port.
  const drake::systems::InputPort<double>& get_gps_position_input_port() const {
    return get_input_port(gps_position_port_);
  }

  /// Returns the GPS valid flag input port.
  const drake::systems::InputPort<double>& get_gps_valid_input_port() const {
    return get_input_port(gps_valid_port_);
  }

  /// Returns the estimated state output port [px, py, pz, vx, vy, vz].
  const drake::systems::OutputPort<double>& get_estimated_state_output_port() const {
    return get_output_port(state_output_port_);
  }

  /// Returns the covariance diagonal output port.
  const drake::systems::OutputPort<double>& get_covariance_output_port() const {
    return get_output_port(covariance_output_port_);
  }

  /// Set initial state estimate.
  void SetInitialState(drake::systems::Context<double>* context,
                       const Eigen::Vector3d& position,
                       const Eigen::Vector3d& velocity) const;

 private:
  static constexpr int kStateSize = 6;  // [p, v]
  static constexpr int kMeasurementSize = 3;  // GPS [x, y, z]

  // Discrete update: prediction + measurement update
  drake::systems::EventStatus UpdateEstimate(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculations
  void CalcEstimatedState(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;

  void CalcCovariance(const drake::systems::Context<double>& context,
                      drake::systems::BasicVector<double>* output) const;

  // EKF matrices
  Eigen::Matrix<double, kStateSize, kStateSize> F_;  // State transition
  Eigen::Matrix<double, kStateSize, kStateSize> Q_;  // Process noise
  Eigen::Matrix<double, kMeasurementSize, kStateSize> H_;  // Measurement matrix
  Eigen::Matrix<double, kMeasurementSize, kMeasurementSize> R_;  // Measurement noise

  // Time step
  double dt_;

  // Port indices
  int gps_position_port_{-1};
  int gps_valid_port_{-1};
  int state_output_port_{-1};
  int covariance_output_port_{-1};

  // Discrete state indices
  // State is stored as: [x(6), P_flat(36)] = 42 elements
  int state_index_{-1};
};

}  // namespace quad_rope_lift
