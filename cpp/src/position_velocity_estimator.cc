#include "position_velocity_estimator.h"

#include <iostream>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

PositionVelocityEstimator::PositionVelocityEstimator(
    double dt, const EstimatorParams& params)
    : dt_(dt) {

  // Build state transition matrix F (constant velocity model)
  // x[k+1] = F * x[k]
  // [p]   [I  dt*I] [p]
  // [v] = [0   I  ] [v]
  F_.setIdentity();
  F_.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();

  // Build process noise covariance Q
  Q_.setZero();
  Q_.block<3, 3>(0, 0) = params.position_process_noise.asDiagonal();
  Q_.block<3, 3>(3, 3) = params.velocity_process_noise.asDiagonal();
  Q_ *= dt;  // Scale by time step

  // Build measurement matrix H (GPS measures position only)
  // z = H * x = [I 0] * [p; v] = p
  H_.setZero();
  H_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // Build measurement noise covariance R
  R_ = params.gps_measurement_noise.asDiagonal();

  // Input ports
  gps_position_port_ = DeclareVectorInputPort(
      "gps_position", BasicVector<double>(kMeasurementSize))
      .get_index();

  gps_valid_port_ = DeclareVectorInputPort(
      "gps_valid", BasicVector<double>(1))
      .get_index();

  // Discrete state: [x(6), P_flat(36)]
  // Initialize with zeros (will be set properly via SetInitialState)
  Eigen::VectorXd initial_state(kStateSize + kStateSize * kStateSize);
  initial_state.setZero();

  // Set initial covariance (diagonal)
  Eigen::Matrix<double, kStateSize, kStateSize> P0;
  P0.setZero();
  P0.block<3, 3>(0, 0) = params.initial_position_variance * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = params.initial_velocity_variance * Eigen::Matrix3d::Identity();

  // Flatten P into state vector
  for (int i = 0; i < kStateSize; ++i) {
    for (int j = 0; j < kStateSize; ++j) {
      initial_state[kStateSize + i * kStateSize + j] = P0(i, j);
    }
  }

  state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update
  DeclarePeriodicDiscreteUpdateEvent(dt, 0.0,
      &PositionVelocityEstimator::UpdateEstimate);

  // Output ports
  state_output_port_ = DeclareVectorOutputPort(
      "estimated_state", BasicVector<double>(kStateSize),
      &PositionVelocityEstimator::CalcEstimatedState)
      .get_index();

  covariance_output_port_ = DeclareVectorOutputPort(
      "covariance_diagonal", BasicVector<double>(kStateSize),
      &PositionVelocityEstimator::CalcCovariance)
      .get_index();
}

void PositionVelocityEstimator::SetInitialState(
    Context<double>* context,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity) const {

  auto& state = context->get_mutable_discrete_state(state_index_);

  // Set position
  state[0] = position.x();
  state[1] = position.y();
  state[2] = position.z();

  // Set velocity
  state[3] = velocity.x();
  state[4] = velocity.y();
  state[5] = velocity.z();
}

EventStatus PositionVelocityEstimator::UpdateEstimate(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {

  // Get current state estimate and covariance
  const auto& state_vec = context.get_discrete_state(state_index_);

  Eigen::Matrix<double, kStateSize, 1> x;
  for (int i = 0; i < kStateSize; ++i) {
    x[i] = state_vec[i];
  }

  Eigen::Matrix<double, kStateSize, kStateSize> P;
  for (int i = 0; i < kStateSize; ++i) {
    for (int j = 0; j < kStateSize; ++j) {
      P(i, j) = state_vec[kStateSize + i * kStateSize + j];
    }
  }

  // === PREDICTION STEP ===
  // x_pred = F * x
  Eigen::Matrix<double, kStateSize, 1> x_pred = F_ * x;

  // P_pred = F * P * F' + Q
  Eigen::Matrix<double, kStateSize, kStateSize> P_pred = F_ * P * F_.transpose() + Q_;

  // === MEASUREMENT UPDATE STEP ===
  // Only update if GPS is valid
  const double gps_valid = get_input_port(gps_valid_port_).Eval(context)[0];

  if (gps_valid > 0.5) {
    // Get GPS measurement
    const auto& gps_meas = get_input_port(gps_position_port_).Eval(context);
    Eigen::Matrix<double, kMeasurementSize, 1> z;
    z << gps_meas[0], gps_meas[1], gps_meas[2];

    // Innovation (measurement residual)
    // y = z - H * x_pred
    Eigen::Matrix<double, kMeasurementSize, 1> y = z - H_ * x_pred;

    // Innovation covariance
    // S = H * P_pred * H' + R
    Eigen::Matrix<double, kMeasurementSize, kMeasurementSize> S =
        H_ * P_pred * H_.transpose() + R_;

    // Kalman gain
    // K = P_pred * H' * S^{-1}
    Eigen::Matrix<double, kStateSize, kMeasurementSize> K =
        P_pred * H_.transpose() * S.inverse();

    // Update state estimate
    // x = x_pred + K * y
    x = x_pred + K * y;

    // Update covariance
    // P = (I - K * H) * P_pred
    Eigen::Matrix<double, kStateSize, kStateSize> I_KH =
        Eigen::Matrix<double, kStateSize, kStateSize>::Identity() - K * H_;
    P = I_KH * P_pred;

    // Ensure symmetry
    P = 0.5 * (P + P.transpose());

  } else {
    // No GPS update, just use prediction
    x = x_pred;
    P = P_pred;
  }

  // Store updated state and covariance
  auto& out_state = discrete_state->get_mutable_vector(state_index_);
  for (int i = 0; i < kStateSize; ++i) {
    out_state[i] = x[i];
  }
  for (int i = 0; i < kStateSize; ++i) {
    for (int j = 0; j < kStateSize; ++j) {
      out_state[kStateSize + i * kStateSize + j] = P(i, j);
    }
  }

  return EventStatus::Succeeded();
}

void PositionVelocityEstimator::CalcEstimatedState(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state = context.get_discrete_state(state_index_);
  for (int i = 0; i < kStateSize; ++i) {
    output->SetAtIndex(i, state[i]);
  }
}

void PositionVelocityEstimator::CalcCovariance(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state = context.get_discrete_state(state_index_);

  // Extract diagonal of P
  for (int i = 0; i < kStateSize; ++i) {
    output->SetAtIndex(i, state[kStateSize + i * kStateSize + i]);
  }
}

}  // namespace quad_rope_lift
