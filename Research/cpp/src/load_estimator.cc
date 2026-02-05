#include "load_estimator.h"

#include <cmath>
#include <iostream>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

LoadStateEstimator::LoadStateEstimator(int num_cables, double dt,
                                       const LoadEstimatorParams& params)
    : num_cables_(num_cables),
      R_cable_(params.cable_constraint_noise * params.cable_constraint_noise),
      tension_threshold_(params.tension_threshold),
      dt_(dt) {

  // Build state transition matrix F (constant velocity model)
  F_.setIdentity();
  F_.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();

  // Build process noise covariance Q
  Q_.setZero();
  Q_.block<3, 3>(0, 0) = params.position_process_noise.asDiagonal();
  Q_.block<3, 3>(3, 3) = params.velocity_process_noise.asDiagonal();
  Q_ *= dt;

  // GPS measurement matrix: z_gps = [I 0] * [p; v] = p
  H_gps_.setZero();
  H_gps_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // GPS measurement noise
  R_gps_ = params.gps_measurement_noise.asDiagonal();

  // Input ports
  gps_position_port_ = DeclareVectorInputPort(
      "load_gps_position", BasicVector<double>(3))
      .get_index();

  gps_valid_port_ = DeclareVectorInputPort(
      "load_gps_valid", BasicVector<double>(1))
      .get_index();

  quad_positions_port_ = DeclareVectorInputPort(
      "quad_attachment_positions", BasicVector<double>(3 * num_cables))
      .get_index();

  tensions_port_ = DeclareVectorInputPort(
      "cable_tensions", BasicVector<double>(num_cables))
      .get_index();

  cable_lengths_port_ = DeclareVectorInputPort(
      "cable_lengths", BasicVector<double>(num_cables))
      .get_index();

  // Discrete state: [x(6), P_flat(36)]
  Eigen::VectorXd initial_state(kStateSize + kStateSize * kStateSize);
  initial_state.setZero();

  // Set initial covariance
  Eigen::Matrix<double, kStateSize, kStateSize> P0;
  P0.setZero();
  P0.block<3, 3>(0, 0) = params.initial_position_variance * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = params.initial_velocity_variance * Eigen::Matrix3d::Identity();

  for (int i = 0; i < kStateSize; ++i) {
    for (int j = 0; j < kStateSize; ++j) {
      initial_state[kStateSize + i * kStateSize + j] = P0(i, j);
    }
  }

  state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update
  DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, &LoadStateEstimator::UpdateEstimate);

  // Output ports
  state_output_port_ = DeclareVectorOutputPort(
      "estimated_state", BasicVector<double>(kStateSize),
      &LoadStateEstimator::CalcEstimatedState)
      .get_index();

  covariance_output_port_ = DeclareVectorOutputPort(
      "covariance_diagonal", BasicVector<double>(kStateSize),
      &LoadStateEstimator::CalcCovariance)
      .get_index();
}

void LoadStateEstimator::SetInitialState(
    Context<double>* context,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity) const {

  auto& state = context->get_mutable_discrete_state(state_index_);
  state[0] = position.x();
  state[1] = position.y();
  state[2] = position.z();
  state[3] = velocity.x();
  state[4] = velocity.y();
  state[5] = velocity.z();
}

EventStatus LoadStateEstimator::UpdateEstimate(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {

  // Get current state and covariance
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
  x = F_ * x;
  P = F_ * P * F_.transpose() + Q_;

  // === GPS MEASUREMENT UPDATE ===
  const double gps_valid = get_input_port(gps_valid_port_).Eval(context)[0];
  if (gps_valid > 0.5) {
    const auto& gps_meas = get_input_port(gps_position_port_).Eval(context);
    Eigen::Vector3d z_gps(gps_meas[0], gps_meas[1], gps_meas[2]);
    ApplyGpsUpdate(z_gps, x, P);
  }

  // === CABLE CONSTRAINT UPDATES (only when taut) ===
  const auto& tensions = get_input_port(tensions_port_).Eval(context);
  const auto& quad_positions = get_input_port(quad_positions_port_).Eval(context);
  const auto& cable_lengths = get_input_port(cable_lengths_port_).Eval(context);

  for (int i = 0; i < num_cables_; ++i) {
    const double tension = tensions[i];

    if (tension >= tension_threshold_) {
      // Cable is taut - apply constraint
      Eigen::Vector3d quad_attach(
          quad_positions[3 * i],
          quad_positions[3 * i + 1],
          quad_positions[3 * i + 2]);
      double cable_length = cable_lengths[i];

      ApplyCableConstraintUpdate(quad_attach, cable_length, x, P);
    }
    // If slack, do nothing - no constraint update
  }

  // Ensure covariance symmetry
  P = 0.5 * (P + P.transpose());

  // Store updated state
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

void LoadStateEstimator::ApplyGpsUpdate(
    const Eigen::Vector3d& gps_measurement,
    Eigen::Matrix<double, kStateSize, 1>& x,
    Eigen::Matrix<double, kStateSize, kStateSize>& P) const {

  // Innovation
  Eigen::Vector3d y = gps_measurement - H_gps_ * x;

  // Innovation covariance
  Eigen::Matrix3d S = H_gps_ * P * H_gps_.transpose() + R_gps_;

  // Kalman gain
  Eigen::Matrix<double, kStateSize, 3> K = P * H_gps_.transpose() * S.inverse();

  // State update
  x = x + K * y;

  // Covariance update
  Eigen::Matrix<double, kStateSize, kStateSize> I_KH =
      Eigen::Matrix<double, kStateSize, kStateSize>::Identity() - K * H_gps_;
  P = I_KH * P;
}

void LoadStateEstimator::ApplyCableConstraintUpdate(
    const Eigen::Vector3d& quad_attachment,
    double cable_length,
    Eigen::Matrix<double, kStateSize, 1>& x,
    Eigen::Matrix<double, kStateSize, kStateSize>& P) const {

  // Current estimated load position
  Eigen::Vector3d p_load = x.head<3>();

  // Vector from quad attachment to load
  Eigen::Vector3d delta = p_load - quad_attachment;
  double distance = delta.norm();

  // Avoid division by zero
  if (distance < 1e-6) {
    return;
  }

  // Cable constraint: h(x) = ||p_load - p_quad|| - L = 0
  // Measurement is the "expected" value = 0 (constraint should be satisfied)
  // Predicted measurement = distance - cable_length
  double h = distance - cable_length;

  // Jacobian: dh/dx = [d(distance)/dp, 0(3x1)]
  // d(distance)/dp = (p_load - p_quad) / ||p_load - p_quad||
  Eigen::Matrix<double, 1, kStateSize> H_cable;
  H_cable.setZero();
  H_cable.block<1, 3>(0, 0) = delta.transpose() / distance;

  // Innovation: z - h(x) = 0 - h = -h
  double y = -h;

  // Innovation covariance (scalar)
  double S = (H_cable * P * H_cable.transpose())(0, 0) + R_cable_;

  // Kalman gain
  Eigen::Matrix<double, kStateSize, 1> K = P * H_cable.transpose() / S;

  // State update
  x = x + K * y;

  // Covariance update
  Eigen::Matrix<double, kStateSize, kStateSize> I_KH =
      Eigen::Matrix<double, kStateSize, kStateSize>::Identity() - K * H_cable;
  P = I_KH * P;
}

void LoadStateEstimator::CalcEstimatedState(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state = context.get_discrete_state(state_index_);
  for (int i = 0; i < kStateSize; ++i) {
    output->SetAtIndex(i, state[i]);
  }
}

void LoadStateEstimator::CalcCovariance(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state = context.get_discrete_state(state_index_);
  for (int i = 0; i < kStateSize; ++i) {
    output->SetAtIndex(i, state[kStateSize + i * kStateSize + i]);
  }
}

}  // namespace quad_rope_lift
