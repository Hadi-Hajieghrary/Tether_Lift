#pragma once

#include <Eigen/Core>
#include <deque>

#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the decentralized adaptive load mass estimator.
struct AdaptiveEstimatorParams {
  /// Adaptation gain (larger = faster convergence, but more noise sensitivity)
  double gamma = 0.5;

  /// Composite sliding variable gain (λ in s = ė + λe)
  double lambda = 1.0;

  /// Concurrent learning weight (ρ > 0 enables historical data)
  double rho = 0.5;

  /// Maximum number of stored data points for concurrent learning
  int max_history_size = 50;

  /// Minimum singular value increase to store new data point
  double sigma_threshold = 0.1;

  /// Minimum theta estimate (prevents division issues)
  double theta_min = 0.1;

  /// Maximum theta estimate (reasonable upper bound)
  double theta_max = 50.0;

  /// Initial theta estimate [kg] (conservative guess for m_L/N)
  double initial_theta = 1.0;

  /// Low-pass filter time constant for acceleration estimate [s]
  double accel_filter_tau = 0.1;

  /// Minimum excitation threshold for data recording
  double min_excitation = 0.5;
};

/// Data point for concurrent learning history buffer.
struct ConcurrentLearningDataPoint {
  double time;        ///< Time of recording
  double Y;           ///< Regressor value
  double phi;         ///< Measured output (filtered acceleration · tension_dir)
};

/// Decentralized adaptive estimator for load mass share (θ = m_L/N).
///
/// Each drone runs an identical copy of this estimator, using only:
/// - Its own cable tension measurement T_i
/// - Load position/velocity from local GPS
/// - Estimated load acceleration (derived from velocity)
///
/// The key insight is that in symmetric configurations:
///   T_i ≈ (m_L/N) * ||g + a_L|| / cos(φ_i)
///
/// So θ̂_i = T_i * cos(φ_i) / ||g + a_L|| converges to m_L/N.
///
/// Uses concurrent learning (Chowdhary & Johnson 2011) to achieve
/// convergence without persistence of excitation.
///
/// Input ports:
///   - cable_tension: Measured cable tension magnitude [N] (1)
///   - cable_direction: Unit vector from quad to load in world frame (3)
///   - load_position: Estimated load position [m] (3)
///   - load_velocity: Estimated load velocity [m/s] (3)
///   - load_position_des: Desired load position [m] (3)
///   - load_velocity_des: Desired load velocity [m/s] (3)
///
/// Output ports:
///   - theta_hat: Estimated load mass share m_L/N [kg] (1)
///   - estimated_weight_share: θ̂ * g [N] - feedforward for controller (1)
///
class AdaptiveLoadEstimator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AdaptiveLoadEstimator);

  /// Constructs the adaptive load mass estimator.
  ///
  /// @param dt Update period [s]
  /// @param params Estimator parameters
  AdaptiveLoadEstimator(double dt,
                        const AdaptiveEstimatorParams& params = AdaptiveEstimatorParams());

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_tension_input_port() const {
    return get_input_port(tension_port_);
  }
  const drake::systems::InputPort<double>& get_cable_direction_input_port() const {
    return get_input_port(cable_direction_port_);
  }
  const drake::systems::InputPort<double>& get_load_position_input_port() const {
    return get_input_port(load_position_port_);
  }
  const drake::systems::InputPort<double>& get_load_velocity_input_port() const {
    return get_input_port(load_velocity_port_);
  }
  const drake::systems::InputPort<double>& get_load_position_des_input_port() const {
    return get_input_port(load_position_des_port_);
  }
  const drake::systems::InputPort<double>& get_load_velocity_des_input_port() const {
    return get_input_port(load_velocity_des_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_theta_hat_output_port() const {
    return get_output_port(theta_hat_port_);
  }
  const drake::systems::OutputPort<double>& get_weight_share_output_port() const {
    return get_output_port(weight_share_port_);
  }

  /// Set initial theta estimate.
  void SetInitialTheta(drake::systems::Context<double>* context,
                       double theta_init) const;

  /// Get current theta estimate.
  double GetThetaHat(const drake::systems::Context<double>& context) const;

  /// Get number of stored data points.
  int GetHistorySize(const drake::systems::Context<double>& context) const;

  /// Returns the parameters.
  const AdaptiveEstimatorParams& params() const { return params_; }

 private:
  // Unrestricted update: compute regressor and update theta estimate
  // Uses unrestricted update to modify both discrete and abstract state
  drake::systems::EventStatus UpdateEstimate(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const;

  // Output calculations
  void CalcThetaHat(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;
  void CalcWeightShare(const drake::systems::Context<double>& context,
                       drake::systems::BasicVector<double>* output) const;

  // Check if data point is informative for concurrent learning
  bool IsDataPointInformative(
      double Y_new,
      const std::deque<ConcurrentLearningDataPoint>& history) const;

  // Parameters
  AdaptiveEstimatorParams params_;
  double dt_;

  // Gravity
  static constexpr double kGravity = 9.81;

  // Port indices
  drake::systems::InputPortIndex tension_port_;
  drake::systems::InputPortIndex cable_direction_port_;
  drake::systems::InputPortIndex load_position_port_;
  drake::systems::InputPortIndex load_velocity_port_;
  drake::systems::InputPortIndex load_position_des_port_;
  drake::systems::InputPortIndex load_velocity_des_port_;

  drake::systems::OutputPortIndex theta_hat_port_;
  drake::systems::OutputPortIndex weight_share_port_;

  // Discrete state indices
  // State layout:
  //   [theta_hat(1), prev_load_vel(3), filtered_accel(3),
  //    history_count(1), history_data(max_size * 3)]
  drake::systems::DiscreteStateIndex state_index_;

  // State offsets
  static constexpr int kThetaOffset = 0;
  static constexpr int kPrevVelOffset = 1;
  static constexpr int kFilteredAccelOffset = 4;
  static constexpr int kHistoryCountOffset = 7;
  static constexpr int kHistoryDataOffset = 8;

  // Abstract state for history buffer (cleaner than packing in discrete state)
  drake::systems::AbstractStateIndex history_index_;
};

}  // namespace quad_rope_lift
