#include "decentralized_load_estimator.h"

#include <algorithm>
#include <cmath>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

DecentralizedLoadEstimator::DecentralizedLoadEstimator(
    double dt, const DecentralizedLoadEstimatorParams& params)
    : dt_(dt), params_(params) {

  // Input ports
  quad_position_port_ =
      DeclareVectorInputPort("quad_position", BasicVector<double>(3))
          .get_index();
  quad_velocity_port_ =
      DeclareVectorInputPort("quad_velocity", BasicVector<double>(3))
          .get_index();
  cable_direction_port_ =
      DeclareVectorInputPort("cable_direction", BasicVector<double>(3))
          .get_index();
  cable_length_port_ =
      DeclareVectorInputPort("cable_length", BasicVector<double>(1))
          .get_index();
  cable_tension_port_ =
      DeclareVectorInputPort("cable_tension", BasicVector<double>(1))
          .get_index();

  // Initialize discrete state
  // [p_L(3), v_L(3), P_diag(6)]
  Eigen::VectorXd initial_state(kStateSize);
  initial_state.setZero();

  // Initialize covariance diagonal (high uncertainty initially)
  initial_state.segment<3>(kCovarianceOffset).setConstant(10.0);      // position variance
  initial_state.segment<3>(kCovarianceOffset + 3).setConstant(1.0);   // velocity variance

  state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update
  DeclarePeriodicDiscreteUpdateEvent(dt_, 0.0,
                                     &DecentralizedLoadEstimator::UpdateEstimate);

  // Output ports
  load_position_port_ =
      DeclareVectorOutputPort("load_position_estimate", BasicVector<double>(3),
                              &DecentralizedLoadEstimator::CalcLoadPosition)
          .get_index();
  load_velocity_port_ =
      DeclareVectorOutputPort("load_velocity_estimate", BasicVector<double>(3),
                              &DecentralizedLoadEstimator::CalcLoadVelocity)
          .get_index();
  covariance_port_ =
      DeclareVectorOutputPort("estimation_covariance", BasicVector<double>(6),
                              &DecentralizedLoadEstimator::CalcCovariance)
          .get_index();
}

EventStatus DecentralizedLoadEstimator::UpdateEstimate(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {

  // Get current state
  const auto& state = context.get_discrete_state(state_index_).value();
  Eigen::Vector3d p_L_hat = state.segment<3>(kPositionOffset);
  Eigen::Vector3d v_L_hat = state.segment<3>(kVelocityOffset);
  Eigen::Vector<double, 6> P_diag = state.segment<6>(kCovarianceOffset);

  // Get inputs
  const auto& pq_raw = get_input_port(quad_position_port_).Eval(context);
  Eigen::Vector3d p_Q(pq_raw[0], pq_raw[1], pq_raw[2]);

  const auto& vq_raw = get_input_port(quad_velocity_port_).Eval(context);
  Eigen::Vector3d v_Q(vq_raw[0], vq_raw[1], vq_raw[2]);

  const auto& n_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d n(n_raw[0], n_raw[1], n_raw[2]);

  // Normalize cable direction (defensive)
  double n_norm = n.norm();
  if (n_norm > 1e-6) {
    n /= n_norm;
  } else {
    n = Eigen::Vector3d(0, 0, -1);  // Default: straight down
  }

  const double L = get_input_port(cable_length_port_).Eval(context)[0];
  const double T = get_input_port(cable_tension_port_).Eval(context)[0];

  // === PREDICTION STEP ===
  // Simple kinematic prediction: p_L = p_L + v_L * dt
  p_L_hat += v_L_hat * dt_;

  // Velocity prediction with damping toward quasi-static estimate
  // v_L_predicted = (1 - damping) * v_L + damping * v_quasi_static
  // For quasi-static: v_L ≈ v_Q (load follows quad approximately)
  v_L_hat = (1.0 - params_.velocity_damping) * v_L_hat +
            params_.velocity_damping * v_Q;

  // Process noise: increase covariance
  double pos_var_increase = params_.position_process_noise * params_.position_process_noise * dt_;
  double vel_var_increase = params_.velocity_process_noise * params_.velocity_process_noise * dt_;
  P_diag.head<3>().array() += pos_var_increase;
  P_diag.tail<3>().array() += vel_var_increase;

  // === MEASUREMENT UPDATE ===
  // Geometric measurement: p_L_measured = p_Q - L * n
  // where n points from quad toward load attachment
  Eigen::Vector3d p_L_measured = p_Q - L * n;

  // Innovation (measurement residual)
  Eigen::Vector3d innovation = p_L_measured - p_L_hat;

  // Measurement noise (position estimate from cable geometry)
  // Depends on cable direction uncertainty and length uncertainty
  double meas_var_base = params_.catenary_position_noise * params_.catenary_position_noise;
  double dir_var = params_.cable_direction_noise * params_.cable_direction_noise * L * L;
  double len_var = params_.cable_length_noise * params_.cable_length_noise;
  double meas_variance = meas_var_base + dir_var + len_var;

  // Tension-based confidence weighting
  // Higher tension = straighter cable = more reliable direction measurement
  double tension_confidence = 1.0;
  if (T > 0.1) {
    // Confidence increases with tension (cables are straighter under load)
    tension_confidence = std::min(1.0, T / 20.0);  // Full confidence at ~20 N
  }
  meas_variance /= (0.1 + 0.9 * tension_confidence);

  // Outlier rejection: check innovation magnitude
  double innovation_magnitude = innovation.norm();
  double expected_std = std::sqrt(P_diag.head<3>().sum() / 3.0 + meas_variance);

  bool valid_measurement =
      (innovation_magnitude < params_.max_innovation_sigma * expected_std);

  if (valid_measurement) {
    // Simplified scalar Kalman gain per dimension
    // K_i = P_i / (P_i + R)
    for (int i = 0; i < 3; ++i) {
      double K = (P_diag[i] * params_.cable_trust_factor) /
                 (P_diag[i] + meas_variance);
      p_L_hat[i] += K * innovation[i];
      P_diag[i] *= (1.0 - K);
    }

    // Velocity update from position change
    // This is a simplified approach; full implementation would track
    // position history for numerical differentiation
    double alpha = 0.3;  // Blending factor
    Eigen::Vector3d implied_velocity = innovation / dt_;
    v_L_hat = (1.0 - alpha) * v_L_hat + alpha * implied_velocity;
  }

  // Ensure covariance bounds
  for (int i = 0; i < 6; ++i) {
    P_diag[i] = std::max(1e-6, std::min(100.0, P_diag[i]));
  }

  // === Store updated state ===
  auto& out_state = discrete_state->get_mutable_vector(state_index_);
  for (int i = 0; i < 3; ++i) {
    out_state[kPositionOffset + i] = p_L_hat[i];
    out_state[kVelocityOffset + i] = v_L_hat[i];
  }
  for (int i = 0; i < 6; ++i) {
    out_state[kCovarianceOffset + i] = P_diag[i];
  }

  return EventStatus::Succeeded();
}

void DecentralizedLoadEstimator::CalcLoadPosition(
    const Context<double>& context,
    BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetFromVector(state.segment<3>(kPositionOffset));
}

void DecentralizedLoadEstimator::CalcLoadVelocity(
    const Context<double>& context,
    BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetFromVector(state.segment<3>(kVelocityOffset));
}

void DecentralizedLoadEstimator::CalcCovariance(
    const Context<double>& context,
    BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetFromVector(state.segment<6>(kCovarianceOffset));
}

void DecentralizedLoadEstimator::SetInitialState(
    Context<double>* context,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity) const {
  auto& state = context->get_mutable_discrete_state(state_index_);
  for (int i = 0; i < 3; ++i) {
    state[kPositionOffset + i] = position[i];
    state[kVelocityOffset + i] = velocity[i];
  }
}

Eigen::Vector3d DecentralizedLoadEstimator::GetLoadPosition(
    const Context<double>& context) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  return state.segment<3>(kPositionOffset);
}

Eigen::Vector3d DecentralizedLoadEstimator::GetLoadVelocity(
    const Context<double>& context) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  return state.segment<3>(kVelocityOffset);
}

}  // namespace quad_rope_lift
