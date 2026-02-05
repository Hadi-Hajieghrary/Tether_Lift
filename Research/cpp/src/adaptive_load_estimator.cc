#include "adaptive_load_estimator.h"

#include <algorithm>
#include <cmath>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;
using drake::systems::State;

AdaptiveLoadEstimator::AdaptiveLoadEstimator(
    double dt, const AdaptiveEstimatorParams& params)
    : dt_(dt), params_(params) {

  // Input ports
  tension_port_ =
      DeclareVectorInputPort("cable_tension", BasicVector<double>(1))
          .get_index();
  cable_direction_port_ =
      DeclareVectorInputPort("cable_direction", BasicVector<double>(3))
          .get_index();
  load_position_port_ =
      DeclareVectorInputPort("load_position", BasicVector<double>(3))
          .get_index();
  load_velocity_port_ =
      DeclareVectorInputPort("load_velocity", BasicVector<double>(3))
          .get_index();
  load_position_des_port_ =
      DeclareVectorInputPort("load_position_des", BasicVector<double>(3))
          .get_index();
  load_velocity_des_port_ =
      DeclareVectorInputPort("load_velocity_des", BasicVector<double>(3))
          .get_index();

  // Initialize discrete state
  // [theta_hat(1), prev_load_vel(3), filtered_accel(3)]
  constexpr int kDiscreteStateSize = 7;
  Eigen::VectorXd initial_state(kDiscreteStateSize);
  initial_state.setZero();
  initial_state[kThetaOffset] = params_.initial_theta;

  state_index_ = DeclareDiscreteState(initial_state);

  // Abstract state for history buffer
  history_index_ = DeclareAbstractState(
      drake::Value<std::deque<ConcurrentLearningDataPoint>>());

  // Use unrestricted update to modify both discrete and abstract state
  DeclarePeriodicUnrestrictedUpdateEvent(dt_, 0.0,
                                         &AdaptiveLoadEstimator::UpdateEstimate);

  // Output ports
  theta_hat_port_ =
      DeclareVectorOutputPort("theta_hat", BasicVector<double>(1),
                              &AdaptiveLoadEstimator::CalcThetaHat)
          .get_index();

  weight_share_port_ =
      DeclareVectorOutputPort("estimated_weight_share", BasicVector<double>(1),
                              &AdaptiveLoadEstimator::CalcWeightShare)
          .get_index();
}

EventStatus AdaptiveLoadEstimator::UpdateEstimate(
    const Context<double>& context,
    State<double>* state) const {
  // Get current discrete state
  const auto& discrete_state = context.get_discrete_state(state_index_).value();
  double theta_hat = discrete_state[kThetaOffset];
  Eigen::Vector3d prev_vel = discrete_state.segment<3>(kPrevVelOffset);
  Eigen::Vector3d filtered_accel = discrete_state.segment<3>(kFilteredAccelOffset);

  // Get history buffer from context
  const auto& history =
      context.get_abstract_state<std::deque<ConcurrentLearningDataPoint>>(
          history_index_);

  // Get inputs
  const double T_i = get_input_port(tension_port_).Eval(context)[0];
  const auto& n_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d n_i(n_raw[0], n_raw[1], n_raw[2]);

  const auto& p_L_raw = get_input_port(load_position_port_).Eval(context);
  Eigen::Vector3d p_L(p_L_raw[0], p_L_raw[1], p_L_raw[2]);

  const auto& v_L_raw = get_input_port(load_velocity_port_).Eval(context);
  Eigen::Vector3d v_L(v_L_raw[0], v_L_raw[1], v_L_raw[2]);

  const auto& p_d_raw = get_input_port(load_position_des_port_).Eval(context);
  Eigen::Vector3d p_d(p_d_raw[0], p_d_raw[1], p_d_raw[2]);

  const auto& v_d_raw = get_input_port(load_velocity_des_port_).Eval(context);
  Eigen::Vector3d v_d(v_d_raw[0], v_d_raw[1], v_d_raw[2]);

  // === Estimate load acceleration (numerical differentiation + filtering) ===
  Eigen::Vector3d raw_accel = (v_L - prev_vel) / dt_;

  // Low-pass filter
  const double alpha = dt_ / (params_.accel_filter_tau + dt_);
  filtered_accel = alpha * raw_accel + (1.0 - alpha) * filtered_accel;

  // === Compute regressor ===
  // From load dynamics: m_L * (a_L + g) = sum_i T_i * n_i
  // For drone i in symmetric config: T_i ≈ (m_L/N) * ||a_L + g|| / cos(φ_i)
  // Regressor: Y_i = ||a_L + g||
  // Measurement: phi = T_i * cos(φ_i) where cos(φ_i) = n_i · e_z / ||n_i||

  Eigen::Vector3d gravity_vec(0.0, 0.0, kGravity);
  Eigen::Vector3d a_plus_g = filtered_accel + gravity_vec;
  double Y_i = a_plus_g.norm();

  // Cable angle from vertical (n_i points from quad to load)
  // cos(phi) = -n_i · e_z (negative because n_i points down when vertical)
  double cos_phi = -n_i.z();  // Assuming n_i is normalized
  cos_phi = std::max(0.1, cos_phi);  // Avoid division issues

  // Measured output: phi = T_i * cos(φ_i) ≈ θ * Y
  double phi = T_i * cos_phi;

  // === Compute sliding variable for adaptation ===
  Eigen::Vector3d e = p_L - p_d;
  Eigen::Vector3d e_dot = v_L - v_d;
  Eigen::Vector3d s = e_dot + params_.lambda * e;

  // Project sliding variable onto cable direction for scalar adaptation
  double s_proj = s.dot(n_i);

  // === Gradient adaptation law ===
  // θ̇ = -γ * Y * s_proj
  double theta_dot = -params_.gamma * Y_i * s_proj;

  // === Concurrent learning term (if enabled) ===
  if (params_.rho > 0.0 && !history.empty()) {
    double cl_term = 0.0;
    for (const auto& data : history) {
      // Prediction error on stored data
      double epsilon_j = data.Y * theta_hat - data.phi;
      cl_term += data.Y * epsilon_j;
    }
    theta_dot -= params_.gamma * params_.rho * cl_term;
  }

  // === Integrate with projection ===
  theta_hat += theta_dot * dt_;
  theta_hat = std::clamp(theta_hat, params_.theta_min, params_.theta_max);

  // === Update mutable state ===
  // Get mutable abstract state for history
  auto& mutable_history =
      state->get_mutable_abstract_state<std::deque<ConcurrentLearningDataPoint>>(
          history_index_);

  // Only store if there's sufficient excitation
  if (Y_i > params_.min_excitation && IsDataPointInformative(Y_i, history)) {
    ConcurrentLearningDataPoint point;
    point.time = context.get_time();
    point.Y = Y_i;
    point.phi = phi;

    mutable_history.push_back(point);

    // Limit buffer size
    while (static_cast<int>(mutable_history.size()) > params_.max_history_size) {
      mutable_history.pop_front();
    }
  }

  // === Store updated discrete state ===
  auto& out_state = state->get_mutable_discrete_state(state_index_);
  out_state[kThetaOffset] = theta_hat;
  out_state[kPrevVelOffset] = v_L.x();
  out_state[kPrevVelOffset + 1] = v_L.y();
  out_state[kPrevVelOffset + 2] = v_L.z();
  out_state[kFilteredAccelOffset] = filtered_accel.x();
  out_state[kFilteredAccelOffset + 1] = filtered_accel.y();
  out_state[kFilteredAccelOffset + 2] = filtered_accel.z();

  return EventStatus::Succeeded();
}

bool AdaptiveLoadEstimator::IsDataPointInformative(
    double Y_new,
    const std::deque<ConcurrentLearningDataPoint>& history) const {

  if (history.empty()) {
    return true;  // First point is always informative
  }

  // Check if new Y differs sufficiently from recent data
  // This is a simplified version of the singular value check

  // Check if adding this point increases information
  // (simplified: check if Y differs from mean by threshold)
  double mean_Y = 0.0;
  for (const auto& data : history) {
    mean_Y += data.Y;
  }
  mean_Y /= history.size();

  return std::abs(Y_new - mean_Y) > params_.sigma_threshold;
}

void AdaptiveLoadEstimator::CalcThetaHat(const Context<double>& context,
                                         BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetAtIndex(0, state[kThetaOffset]);
}

void AdaptiveLoadEstimator::CalcWeightShare(const Context<double>& context,
                                            BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetAtIndex(0, state[kThetaOffset] * kGravity);
}

void AdaptiveLoadEstimator::SetInitialTheta(Context<double>* context,
                                            double theta_init) const {
  auto& state = context->get_mutable_discrete_state(state_index_);
  state[kThetaOffset] = std::clamp(theta_init, params_.theta_min, params_.theta_max);
}

double AdaptiveLoadEstimator::GetThetaHat(const Context<double>& context) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  return state[kThetaOffset];
}

int AdaptiveLoadEstimator::GetHistorySize(const Context<double>& context) const {
  const auto& history =
      context.get_abstract_state<std::deque<ConcurrentLearningDataPoint>>(
          history_index_);
  return static_cast<int>(history.size());
}

}  // namespace quad_rope_lift
