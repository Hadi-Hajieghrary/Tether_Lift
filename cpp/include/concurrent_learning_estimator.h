#pragma once

/// @file concurrent_learning_estimator.h
/// @brief Concurrent Learning for adaptive parameter estimation [GPAC Layer 3]
///
/// Implements the concurrent learning algorithm from the GPAC proposal:
///
/// θ̇ = Γ·(Y^T·s + Σⱼ Yⱼ^T·εⱼ)
///
/// where:
///   - θ is the parameter vector (load mass, etc.)
///   - Y is the regressor matrix
///   - s is the composite variable (tracking error)
///   - (Yⱼ, εⱼ) are stored data points from the history stack
///   - Γ is the adaptation gain matrix
///
/// Key features:
/// - Guaranteed parameter convergence without persistent excitation
/// - History stack with rank-maximizing data selection
/// - Projection to maintain parameter bounds

#include <Eigen/Core>
#include <drake/systems/framework/leaf_system.h>
#include <deque>
#include <vector>

namespace quad_rope_lift {
namespace gpac {

/// @brief Parameters for concurrent learning estimator
struct ConcurrentLearningParams {
  // Adaptation gains
  double gamma_mass = 0.5;           ///< Mass adaptation gain
  double gamma_cable = 0.1;          ///< Cable parameter gain

  // History stack
  int max_history_size = 50;         ///< Maximum data points
  double min_singular_value = 0.01;  ///< Rank check threshold
  double data_spacing_time = 0.1;    ///< Minimum time between samples [s]

  // Parameter bounds
  double mass_min = 0.5;             ///< Minimum mass estimate [kg]
  double mass_max = 20.0;            ///< Maximum mass estimate [kg]

  // Composite variable gains
  double lambda = 1.0;               ///< Composite variable filter gain

  // Initial estimates
  double initial_mass = 3.0;         ///< Initial load mass estimate [kg]
};

/// @brief Data point for history stack
struct DataPoint {
  double timestamp;
  Eigen::VectorXd regressor;    ///< Yⱼ
  Eigen::VectorXd error;        ///< εⱼ
};

/// @brief Concurrent Learning Estimator (standalone class)
///
/// Can be used standalone or wrapped in a Drake LeafSystem.
class ConcurrentLearningCore {
 public:
  explicit ConcurrentLearningCore(const ConcurrentLearningParams& params = ConcurrentLearningParams());

  /// Reset estimator to initial state
  void Reset();

  /// Update estimator with new measurement
  /// @param Y Regressor matrix (n_params x 1 or n_params x n_outputs)
  /// @param s Composite tracking error
  /// @param timestamp Current time
  /// @param dt Time step
  void Update(const Eigen::VectorXd& Y,
              const Eigen::VectorXd& s,
              double timestamp,
              double dt);

  /// Get current parameter estimate
  const Eigen::VectorXd& theta() const { return theta_; }

  /// Get estimated mass
  double mass() const { return theta_(0); }

  /// Get history stack size
  int history_size() const { return static_cast<int>(history_.size()); }

  /// Check if history has sufficient rank
  bool has_sufficient_rank() const { return has_sufficient_rank_; }

  /// Set parameter estimate
  void set_theta(const Eigen::VectorXd& theta) { theta_ = theta; }

 private:
  /// Add data point to history if it improves rank
  void MaybeAddToHistory(const Eigen::VectorXd& Y,
                         const Eigen::VectorXd& error,
                         double timestamp);

  /// Compute concurrent learning term
  Eigen::VectorXd ComputeConcurrentTerm() const;

  /// Project parameter to feasible set
  void ProjectParameters();

  /// Check if adding data point improves rank
  bool ImprovedRank(const Eigen::VectorXd& Y) const;

  ConcurrentLearningParams params_;

  /// Parameter estimates: [mass, (optional: cable params)]
  Eigen::VectorXd theta_;

  /// History stack
  std::deque<DataPoint> history_;

  /// Timestamp of last added data point
  double last_data_time_ = -1.0;

  /// Flag for rank condition
  bool has_sufficient_rank_ = false;
};

/// @brief Concurrent Learning Estimator as Drake LeafSystem
///
/// Input ports:
///   0: regressor (3D): [Y1, Y2, Y3] regressor vector
///   1: tracking_error (3D): [s1, s2, s3] composite error
///   2: control_force (3D): [F1, F2, F3] applied force (for regressor)
///   3: acceleration (3D): [a1, a2, a3] measured acceleration
///
/// Output ports:
///   0: theta (1D or more): estimated parameters [mass, ...]
///   1: mass (1D): estimated load mass
///
/// State:
///   Continuous: parameter vector θ
class ConcurrentLearningEstimator final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConcurrentLearningEstimator);

  explicit ConcurrentLearningEstimator(
      const ConcurrentLearningParams& params = ConcurrentLearningParams());

  // === Input port accessors ===

  /// Composite tracking error s
  const drake::systems::InputPort<double>& get_tracking_error_input() const {
    return get_input_port(tracking_error_port_);
  }

  /// Control force F (for building regressor)
  const drake::systems::InputPort<double>& get_control_force_input() const {
    return get_input_port(control_force_port_);
  }

  /// Measured acceleration (for building regressor)
  const drake::systems::InputPort<double>& get_acceleration_input() const {
    return get_input_port(acceleration_port_);
  }

  // === Output port accessors ===

  /// Full parameter vector
  const drake::systems::OutputPort<double>& get_theta_output() const {
    return get_output_port(theta_port_);
  }

  /// Estimated mass
  const drake::systems::OutputPort<double>& get_mass_output() const {
    return get_output_port(mass_port_);
  }

  const ConcurrentLearningParams& params() const { return params_; }

 private:
  void DoCalcTimeDerivatives(
      const drake::systems::Context<double>& context,
      drake::systems::ContinuousState<double>* derivatives) const override;

  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  void CalcTheta(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcMass(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  /// Build regressor from current inputs
  Eigen::VectorXd BuildRegressor(
      const drake::systems::Context<double>& context) const;

  ConcurrentLearningParams params_;

  // Mutable history (since Update modifies it)
  mutable ConcurrentLearningCore core_;
  mutable double last_update_time_ = -1.0;

  // Port indices
  int tracking_error_port_{};
  int control_force_port_{};
  int acceleration_port_{};
  int theta_port_{};
  int mass_port_{};
};

}  // namespace gpac
}  // namespace quad_rope_lift
