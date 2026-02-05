#pragma once

#include <Eigen/Core>
#include <vector>

#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the load state estimator.
struct LoadEstimatorParams {
  // Process noise
  Eigen::Vector3d position_process_noise{0.01, 0.01, 0.01};  ///< [m]
  Eigen::Vector3d velocity_process_noise{0.1, 0.1, 0.1};     ///< [m/s]

  // GPS measurement noise
  Eigen::Vector3d gps_measurement_noise{0.02, 0.02, 0.05};   ///< [m]

  // Cable constraint parameters
  double cable_constraint_noise = 0.02;  ///< Range measurement noise [m]
  double tension_threshold = 1.0;         ///< Tension threshold for taut detection [N]

  // Initial covariance
  double initial_position_variance = 0.1;  ///< [m²]
  double initial_velocity_variance = 0.5;  ///< [(m/s)²]
};

/// Load state estimator with GPS and taut-gated cable constraints.
///
/// State: [px, py, pz, vx, vy, vz] (6 states)
///
/// Measurements:
///   1. GPS position (direct measurement when valid)
///   2. Cable range constraints (pseudo-measurements when taut)
///      h_i(x) = ||p_quad_i - p_load|| - cable_length_i = 0
///
/// The cable constraints are only applied when:
///   - Measured tension exceeds the threshold (cable is taut)
///   - This prevents biasing the estimate when cables are slack
///
/// Input ports:
///   - load_gps_position: GPS measurement [x, y, z]
///   - load_gps_valid: Flag indicating if GPS is valid
///   - quad_attachment_positions: Stacked quad attachment positions [x0,y0,z0,x1,y1,z1,...]
///   - cable_tensions: Measured tensions per cable [T0, T1, ...]
///   - cable_lengths: Estimated/known cable lengths [L0, L1, ...]
///
/// Output ports:
///   - estimated_state: [px, py, pz, vx, vy, vz]
///   - covariance_diagonal: Diagonal of P matrix
///
class LoadStateEstimator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LoadStateEstimator);

  /// Constructs the load state estimator.
  ///
  /// @param num_cables Number of cables/quadcopters
  /// @param dt Discrete update period [s]
  /// @param params Estimator parameters
  LoadStateEstimator(int num_cables, double dt,
                     const LoadEstimatorParams& params = LoadEstimatorParams());

  // Input port accessors
  const drake::systems::InputPort<double>& get_gps_position_input_port() const {
    return get_input_port(gps_position_port_);
  }

  const drake::systems::InputPort<double>& get_gps_valid_input_port() const {
    return get_input_port(gps_valid_port_);
  }

  const drake::systems::InputPort<double>& get_quad_positions_input_port() const {
    return get_input_port(quad_positions_port_);
  }

  const drake::systems::InputPort<double>& get_tensions_input_port() const {
    return get_input_port(tensions_port_);
  }

  const drake::systems::InputPort<double>& get_cable_lengths_input_port() const {
    return get_input_port(cable_lengths_port_);
  }

  // Output port accessors
  const drake::systems::OutputPort<double>& get_estimated_state_output_port() const {
    return get_output_port(state_output_port_);
  }

  const drake::systems::OutputPort<double>& get_covariance_output_port() const {
    return get_output_port(covariance_output_port_);
  }

  /// Set initial state estimate.
  void SetInitialState(drake::systems::Context<double>* context,
                       const Eigen::Vector3d& position,
                       const Eigen::Vector3d& velocity) const;

 private:
  static constexpr int kStateSize = 6;  // [p, v]
  static constexpr int kGpsMeasSize = 3;  // GPS [x, y, z]

  // Discrete update
  drake::systems::EventStatus UpdateEstimate(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Apply GPS measurement update
  void ApplyGpsUpdate(
      const Eigen::Vector3d& gps_measurement,
      Eigen::Matrix<double, kStateSize, 1>& x,
      Eigen::Matrix<double, kStateSize, kStateSize>& P) const;

  // Apply single cable constraint update (EKF-style)
  void ApplyCableConstraintUpdate(
      const Eigen::Vector3d& quad_attachment,
      double cable_length,
      Eigen::Matrix<double, kStateSize, 1>& x,
      Eigen::Matrix<double, kStateSize, kStateSize>& P) const;

  // Output calculations
  void CalcEstimatedState(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;

  void CalcCovariance(const drake::systems::Context<double>& context,
                      drake::systems::BasicVector<double>* output) const;

  // Number of cables
  int num_cables_;

  // EKF matrices
  Eigen::Matrix<double, kStateSize, kStateSize> F_;  // State transition
  Eigen::Matrix<double, kStateSize, kStateSize> Q_;  // Process noise
  Eigen::Matrix<double, kGpsMeasSize, kStateSize> H_gps_;  // GPS measurement matrix
  Eigen::Matrix<double, kGpsMeasSize, kGpsMeasSize> R_gps_;  // GPS measurement noise
  double R_cable_;  // Cable constraint noise (scalar)
  double tension_threshold_;

  // Time step
  double dt_;

  // Port indices
  int gps_position_port_{-1};
  int gps_valid_port_{-1};
  int quad_positions_port_{-1};
  int tensions_port_{-1};
  int cable_lengths_port_{-1};
  int state_output_port_{-1};
  int covariance_output_port_{-1};

  // Discrete state index
  int state_index_{-1};
};

}  // namespace quad_rope_lift
