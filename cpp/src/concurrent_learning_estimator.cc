/// @file concurrent_learning_estimator.cc
/// @brief Concurrent Learning estimator implementation.

#include "concurrent_learning_estimator.h"

#include <algorithm>
#include <cmath>
#include <Eigen/SVD>

namespace quad_rope_lift {
namespace gpac {

// =============================================================================
// ConcurrentLearningCore Implementation
// =============================================================================

ConcurrentLearningCore::ConcurrentLearningCore(const ConcurrentLearningParams& params)
    : params_(params) {
  theta_.resize(1);
  theta_(0) = params_.initial_mass;
}

void ConcurrentLearningCore::Reset() {
  theta_(0) = params_.initial_mass;
  history_.clear();
  last_data_time_ = -1.0;
  has_sufficient_rank_ = false;
}

void ConcurrentLearningCore::Update(
    const Eigen::VectorXd& Y,
    const Eigen::VectorXd& s,
    double timestamp,
    double dt) {

  // Compute adaptation law:
  // θ̇ = Γ·(Y^T·s + concurrent_term)

  // Online term: Y^T * s
  Eigen::VectorXd online_term = Y.transpose() * s;

  // Concurrent learning term from history
  Eigen::VectorXd concurrent_term = ComputeConcurrentTerm();

  // Total derivative
  Eigen::VectorXd theta_dot = params_.gamma_mass * (online_term + concurrent_term);

  // Euler integration
  theta_ += theta_dot * dt;

  // Project to bounds
  ProjectParameters();

  // Maybe add to history
  // Build prediction error: ε = Y*θ - measured_output
  // For mass estimation: ε = F/m - a = (F - m*a)/m
  // Simplified: store Y and error for later use
  Eigen::VectorXd error = s;  // Use tracking error as proxy
  MaybeAddToHistory(Y, error, timestamp);
}

void ConcurrentLearningCore::MaybeAddToHistory(
    const Eigen::VectorXd& Y,
    const Eigen::VectorXd& error,
    double timestamp) {

  // Check time spacing
  if (last_data_time_ >= 0.0 &&
      timestamp - last_data_time_ < params_.data_spacing_time) {
    return;
  }

  // Check if data improves rank
  if (history_.size() > 0 && !ImprovedRank(Y)) {
    return;
  }

  // Add data point
  DataPoint dp;
  dp.timestamp = timestamp;
  dp.regressor = Y;
  dp.error = error;
  history_.push_back(dp);
  last_data_time_ = timestamp;

  // Remove oldest if over capacity
  while (history_.size() > static_cast<size_t>(params_.max_history_size)) {
    history_.pop_front();
  }

  // Update rank flag
  has_sufficient_rank_ = (history_.size() >= 3);
}

Eigen::VectorXd ConcurrentLearningCore::ComputeConcurrentTerm() const {
  if (history_.empty()) {
    return Eigen::VectorXd::Zero(theta_.size());
  }

  Eigen::VectorXd term = Eigen::VectorXd::Zero(theta_.size());

  for (const auto& dp : history_) {
    // Prediction error: ε_j = Y_j * θ - measured (approximated by stored error)
    // Here we use the stored error directly
    // Contribution: Y_j^T * ε_j
    term += dp.regressor.transpose() * dp.error;
  }

  return term / std::max(1.0, static_cast<double>(history_.size()));
}

void ConcurrentLearningCore::ProjectParameters() {
  // Project mass to valid range
  theta_(0) = std::max(params_.mass_min, std::min(params_.mass_max, theta_(0)));
}

bool ConcurrentLearningCore::ImprovedRank(const Eigen::VectorXd& Y) const {
  if (history_.empty()) {
    return true;
  }

  // Build regressor matrix from history
  Eigen::MatrixXd Y_hist(history_[0].regressor.size(), history_.size());
  for (size_t j = 0; j < history_.size(); ++j) {
    Y_hist.col(j) = history_[j].regressor;
  }

  // Compute SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_before(Y_hist);
  double min_sv_before = svd_before.singularValues().minCoeff();

  // Add new column
  Eigen::MatrixXd Y_new(Y_hist.rows(), Y_hist.cols() + 1);
  Y_new.leftCols(Y_hist.cols()) = Y_hist;
  Y_new.rightCols(1) = Y;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd_after(Y_new);
  double min_sv_after = svd_after.singularValues().minCoeff();

  // Accept if minimum singular value improves
  return min_sv_after > min_sv_before + params_.min_singular_value * 0.1;
}

// =============================================================================
// ConcurrentLearningEstimator (Drake LeafSystem) Implementation
// =============================================================================

ConcurrentLearningEstimator::ConcurrentLearningEstimator(
    const ConcurrentLearningParams& params)
    : params_(params), core_(params) {

  // Input ports
  tracking_error_port_ = DeclareVectorInputPort("tracking_error", 3).get_index();
  control_force_port_ = DeclareVectorInputPort("control_force", 3).get_index();
  acceleration_port_ = DeclareVectorInputPort("acceleration", 3).get_index();

  // Continuous state: theta (1 element: mass)
  DeclareContinuousState(1);

  // Output ports
  theta_port_ = DeclareVectorOutputPort(
      "theta", 1,
      &ConcurrentLearningEstimator::CalcTheta).get_index();

  mass_port_ = DeclareVectorOutputPort(
      "mass", 1,
      &ConcurrentLearningEstimator::CalcMass).get_index();
}

void ConcurrentLearningEstimator::SetDefaultState(
    const drake::systems::Context<double>& context,
    drake::systems::State<double>* state) const {
  auto& continuous_state = state->get_mutable_continuous_state();
  continuous_state.get_mutable_vector().SetAtIndex(0, params_.initial_mass);
  core_.Reset();
}

Eigen::VectorXd ConcurrentLearningEstimator::BuildRegressor(
    const drake::systems::Context<double>& context) const {

  // For mass estimation, the regressor relates force and acceleration:
  // F = m * a  →  F/m = a  →  regressor Y = a (acceleration)
  // Then θ = m, and Y*θ = m*a should equal F
  //
  // Actually, for CL: Y^T * (measured - Y*θ)
  // Y = [ax, ay, az]^T, θ = m
  // Y*θ = [m*ax, m*ay, m*az]^T
  // measured = F = [Fx, Fy, Fz]^T
  // error = F - m*a

  return get_input_port(acceleration_port_).Eval(context);
}

void ConcurrentLearningEstimator::DoCalcTimeDerivatives(
    const drake::systems::Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {

  const double t = context.get_time();
  const double dt = (last_update_time_ > 0.0) ? (t - last_update_time_) : 0.001;

  // Get inputs
  const Eigen::Vector3d s = get_input_port(tracking_error_port_).Eval(context);
  const Eigen::Vector3d F = get_input_port(control_force_port_).Eval(context);
  const Eigen::Vector3d a = get_input_port(acceleration_port_).Eval(context);

  // Build regressor
  Eigen::VectorXd Y = a;

  // Sync core with Drake state
  const double theta_drake = context.get_continuous_state_vector()[0];
  Eigen::VectorXd theta_vec(1);
  theta_vec(0) = theta_drake;
  core_.set_theta(theta_vec);

  // Update core
  core_.Update(Y, s, t, dt);
  last_update_time_ = t;

  // Compute derivative for Drake state
  // θ̇ = Γ·(Y^T·s + concurrent_term)
  Eigen::VectorXd Y_T_s = Y.transpose() * s;

  // Simplified: just use the mass derivative
  double theta_dot = params_.gamma_mass * Y_T_s(0);

  derivatives->get_mutable_vector().SetAtIndex(0, theta_dot);
}

void ConcurrentLearningEstimator::CalcTheta(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  output->SetAtIndex(0, context.get_continuous_state_vector()[0]);
}

void ConcurrentLearningEstimator::CalcMass(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  double mass = context.get_continuous_state_vector()[0];
  // Clamp to valid range
  mass = std::max(params_.mass_min, std::min(params_.mass_max, mass));
  output->SetAtIndex(0, mass);
}

}  // namespace gpac
}  // namespace quad_rope_lift
