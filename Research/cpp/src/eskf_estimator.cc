#include "eskf_estimator.h"

#include <cmath>
#include <iostream>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

// Nominal state indices
constexpr int kPosIdx = 0;
constexpr int kVelIdx = 3;
constexpr int kQuatIdx = 6;
constexpr int kAccelBiasIdx = 10;
constexpr int kGyroBiasIdx = 13;

// Error state indices
constexpr int kErrPosIdx = 0;
constexpr int kErrVelIdx = 3;
constexpr int kErrAttIdx = 6;
constexpr int kErrAccelBiasIdx = 9;
constexpr int kErrGyroBiasIdx = 12;

EskfEstimator::EskfEstimator(double imu_dt, const EskfParams& params)
    : imu_dt_(imu_dt), params_(params) {
  // Build continuous-time process noise covariance (12x12)
  // Noise inputs: [accel_noise(3), gyro_noise(3), accel_bias_rw(3), gyro_bias_rw(3)]
  Q_continuous_.setZero();
  Q_continuous_.block<3, 3>(0, 0) =
      params_.accel_noise_density.array().square().matrix().asDiagonal();
  Q_continuous_.block<3, 3>(3, 3) =
      params_.gyro_noise_density.array().square().matrix().asDiagonal();
  Q_continuous_.block<3, 3>(6, 6) =
      params_.accel_bias_random_walk.array().square().matrix().asDiagonal();
  Q_continuous_.block<3, 3>(9, 9) =
      params_.gyro_bias_random_walk.array().square().matrix().asDiagonal();

  // Input ports
  accel_port_ =
      DeclareVectorInputPort("accel_measurement", BasicVector<double>(3))
          .get_index();
  gyro_port_ =
      DeclareVectorInputPort("gyro_measurement", BasicVector<double>(3))
          .get_index();
  gps_position_port_ =
      DeclareVectorInputPort("gps_position", BasicVector<double>(3))
          .get_index();
  gps_valid_port_ =
      DeclareVectorInputPort("gps_valid", BasicVector<double>(1)).get_index();
  baro_altitude_port_ =
      DeclareVectorInputPort("baro_altitude", BasicVector<double>(1))
          .get_index();
  baro_valid_port_ =
      DeclareVectorInputPort("baro_valid", BasicVector<double>(1)).get_index();

  // Initialize state: nominal_state(16) + P_flat(225)
  const int total_state_size = kNominalStateSize + kCovarianceSize;
  Eigen::VectorXd initial_state(total_state_size);
  initial_state.setZero();

  // Initial quaternion = identity [w, x, y, z] = [1, 0, 0, 0]
  initial_state[kQuatIdx] = 1.0;

  // Initial covariance
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> P0;
  P0.setZero();
  P0.block<3, 3>(kErrPosIdx, kErrPosIdx) =
      params_.initial_position_stddev.array().square().matrix().asDiagonal();
  P0.block<3, 3>(kErrVelIdx, kErrVelIdx) =
      params_.initial_velocity_stddev.array().square().matrix().asDiagonal();
  P0.block<3, 3>(kErrAttIdx, kErrAttIdx) =
      params_.initial_attitude_stddev.array().square().matrix().asDiagonal();
  P0.block<3, 3>(kErrAccelBiasIdx, kErrAccelBiasIdx) =
      params_.initial_accel_bias_stddev.array().square().matrix().asDiagonal();
  P0.block<3, 3>(kErrGyroBiasIdx, kErrGyroBiasIdx) =
      params_.initial_gyro_bias_stddev.array().square().matrix().asDiagonal();

  // Flatten covariance into state vector
  for (int i = 0; i < kErrorStateSize; ++i) {
    for (int j = 0; j < kErrorStateSize; ++j) {
      initial_state[kNominalStateSize + i * kErrorStateSize + j] = P0(i, j);
    }
  }

  state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update at IMU rate
  DeclarePeriodicDiscreteUpdateEvent(imu_dt_, 0.0,
                                     &EskfEstimator::UpdateEstimate);

  // Output ports
  pose_output_port_ =
      DeclareVectorOutputPort("estimated_pose", BasicVector<double>(7),
                              &EskfEstimator::CalcEstimatedPose)
          .get_index();

  velocity_output_port_ =
      DeclareVectorOutputPort("estimated_velocity", BasicVector<double>(3),
                              &EskfEstimator::CalcEstimatedVelocity)
          .get_index();

  state_output_port_ =
      DeclareVectorOutputPort("estimated_state", BasicVector<double>(6),
                              &EskfEstimator::CalcEstimatedState)
          .get_index();

  biases_output_port_ =
      DeclareVectorOutputPort("estimated_biases", BasicVector<double>(6),
                              &EskfEstimator::CalcEstimatedBiases)
          .get_index();

  covariance_output_port_ =
      DeclareVectorOutputPort("covariance_diagonal",
                              BasicVector<double>(kErrorStateSize),
                              &EskfEstimator::CalcCovariance)
          .get_index();
}

EventStatus EskfEstimator::UpdateEstimate(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  // Get current state
  Eigen::VectorXd nominal_state =
      context.get_discrete_state(state_index_).value().head(kNominalStateSize);
  auto P = ExtractCovariance(context.get_discrete_state(state_index_).value());

  // Get IMU measurements
  const auto& accel_meas_raw = get_input_port(accel_port_).Eval(context);
  const auto& gyro_meas_raw = get_input_port(gyro_port_).Eval(context);
  Eigen::Vector3d accel_meas(accel_meas_raw[0], accel_meas_raw[1],
                             accel_meas_raw[2]);
  Eigen::Vector3d gyro_meas(gyro_meas_raw[0], gyro_meas_raw[1],
                            gyro_meas_raw[2]);

  // === PROPAGATION STEP ===
  Propagate(accel_meas, gyro_meas, imu_dt_, nominal_state, P);

  // === MEASUREMENT UPDATES ===

  // GPS position update
  const double gps_valid = get_input_port(gps_valid_port_).Eval(context)[0];
  if (gps_valid > 0.5) {
    const auto& gps_pos = get_input_port(gps_position_port_).Eval(context);
    Eigen::Vector3d gps_position(gps_pos[0], gps_pos[1], gps_pos[2]);
    UpdateGpsPosition(gps_position, nominal_state, P);
  }

  // Barometer altitude update
  const double baro_valid = get_input_port(baro_valid_port_).Eval(context)[0];
  if (baro_valid > 0.5) {
    const double baro_alt =
        get_input_port(baro_altitude_port_).Eval(context)[0];
    UpdateBaroAltitude(baro_alt, nominal_state, P);
  }

  // Store updated state
  auto& out_state = discrete_state->get_mutable_vector(state_index_);
  for (int i = 0; i < kNominalStateSize; ++i) {
    out_state[i] = nominal_state[i];
  }
  for (int i = 0; i < kErrorStateSize; ++i) {
    for (int j = 0; j < kErrorStateSize; ++j) {
      out_state[kNominalStateSize + i * kErrorStateSize + j] = P(i, j);
    }
  }

  return EventStatus::Succeeded();
}

void EskfEstimator::Propagate(
    const Eigen::Vector3d& accel_meas, const Eigen::Vector3d& gyro_meas,
    double dt, Eigen::VectorXd& nominal_state,
    Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const {
  // Extract current state
  Eigen::Vector3d p = ExtractPosition(nominal_state);
  Eigen::Vector3d v = ExtractVelocity(nominal_state);
  Eigen::Vector4d q = ExtractQuaternion(nominal_state);
  Eigen::Vector3d b_a = ExtractAccelBias(nominal_state);
  Eigen::Vector3d b_g = ExtractGyroBias(nominal_state);

  // Correct measurements for bias
  Eigen::Vector3d accel_corrected = accel_meas - b_a;
  Eigen::Vector3d gyro_corrected = gyro_meas - b_g;

  // Rotation matrix from body to world
  Eigen::Matrix3d R = QuaternionToRotationMatrix(q);

  // === Propagate nominal state ===
  // Position: p_new = p + v*dt + 0.5*a*dt²
  Eigen::Vector3d accel_world = R * accel_corrected + gravity_world_;
  p = p + v * dt + 0.5 * accel_world * dt * dt;

  // Velocity: v_new = v + a*dt
  v = v + accel_world * dt;

  // Quaternion: integrate angular velocity
  // dq = 0.5 * q ⊗ [0, ω]
  Eigen::Vector3d angle = gyro_corrected * dt;
  Eigen::Vector4d dq = QuaternionFromAxisAngle(angle);
  q = QuaternionMultiply(q, dq);
  q.normalize();  // Maintain unit quaternion

  // Biases: unchanged in prediction (random walk)

  // Store propagated nominal state
  nominal_state.segment<3>(kPosIdx) = p;
  nominal_state.segment<3>(kVelIdx) = v;
  nominal_state.segment<4>(kQuatIdx) = q;

  // === Propagate covariance ===
  // Error state dynamics: δx_dot = F*δx + G*w
  // F is the Jacobian of the error state dynamics

  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> F;
  F.setZero();

  // ∂δp/∂δv = I
  F.block<3, 3>(kErrPosIdx, kErrVelIdx) = Eigen::Matrix3d::Identity();

  // ∂δv/∂δθ = -R*[a_corrected]_×
  F.block<3, 3>(kErrVelIdx, kErrAttIdx) =
      -R * SkewSymmetric(accel_corrected);

  // ∂δv/∂δb_a = -R
  F.block<3, 3>(kErrVelIdx, kErrAccelBiasIdx) = -R;

  // ∂δθ/∂δθ = -[ω_corrected]_×
  F.block<3, 3>(kErrAttIdx, kErrAttIdx) =
      -SkewSymmetric(gyro_corrected);

  // ∂δθ/∂δb_g = -I
  F.block<3, 3>(kErrAttIdx, kErrGyroBiasIdx) =
      -Eigen::Matrix3d::Identity();

  // Noise input matrix G (maps noise to error state derivatives)
  Eigen::Matrix<double, kErrorStateSize, 12> G;
  G.setZero();

  // Accelerometer noise affects velocity
  G.block<3, 3>(kErrVelIdx, 0) = -R;

  // Gyro noise affects attitude
  G.block<3, 3>(kErrAttIdx, 3) = -Eigen::Matrix3d::Identity();

  // Accel bias random walk
  G.block<3, 3>(kErrAccelBiasIdx, 6) = Eigen::Matrix3d::Identity();

  // Gyro bias random walk
  G.block<3, 3>(kErrGyroBiasIdx, 9) = Eigen::Matrix3d::Identity();

  // Discrete-time state transition: Phi ≈ I + F*dt
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> Phi =
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>::Identity() +
      F * dt;

  // Discrete-time process noise: Q_d ≈ G * Q_c * G^T * dt
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> Q_d =
      G * Q_continuous_ * G.transpose() * dt;

  // Propagate covariance: P = Phi * P * Phi^T + Q_d
  P = Phi * P * Phi.transpose() + Q_d;

  // Ensure symmetry
  P = 0.5 * (P + P.transpose());
}

void EskfEstimator::UpdateGpsPosition(
    const Eigen::Vector3d& gps_position, Eigen::VectorXd& nominal_state,
    Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const {
  // Measurement model: z = p + v, where v is noise
  // H = [I, 0, 0, 0, 0] (3x15)

  Eigen::Matrix<double, 3, kErrorStateSize> H;
  H.setZero();
  H.block<3, 3>(0, kErrPosIdx) = Eigen::Matrix3d::Identity();

  // Measurement noise covariance
  Eigen::Matrix3d R =
      params_.gps_position_noise.array().square().matrix().asDiagonal();

  // Innovation
  Eigen::Vector3d y =
      gps_position - nominal_state.segment<3>(kPosIdx);

  // Innovation covariance
  Eigen::Matrix3d S = H * P * H.transpose() + R;

  // Kalman gain
  Eigen::Matrix<double, kErrorStateSize, 3> K =
      P * H.transpose() * S.inverse();

  // Error state update
  Eigen::Matrix<double, kErrorStateSize, 1> delta_x = K * y;

  // Inject error and reset
  InjectErrorAndReset(delta_x, nominal_state, P);

  // Update covariance: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T (Joseph form)
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> I_KH =
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>::Identity() -
      K * H;
  P = I_KH * P * I_KH.transpose() + K * R * K.transpose();

  // Ensure symmetry
  P = 0.5 * (P + P.transpose());
}

void EskfEstimator::UpdateBaroAltitude(
    double baro_altitude, Eigen::VectorXd& nominal_state,
    Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const {
  // Measurement model: z = p_z + v
  // H = [0, 0, 1, 0, ..., 0] (1x15)

  Eigen::Matrix<double, 1, kErrorStateSize> H;
  H.setZero();
  H(0, kErrPosIdx + 2) = 1.0;  // z-component of position

  // Measurement noise
  double R = params_.baro_altitude_noise * params_.baro_altitude_noise;

  // Innovation
  double y = baro_altitude - nominal_state[kPosIdx + 2];

  // Innovation covariance
  double S = (H * P * H.transpose())(0, 0) + R;

  // Kalman gain
  Eigen::Matrix<double, kErrorStateSize, 1> K = P * H.transpose() / S;

  // Error state update
  Eigen::Matrix<double, kErrorStateSize, 1> delta_x = K * y;

  // Inject error and reset
  InjectErrorAndReset(delta_x, nominal_state, P);

  // Update covariance
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> I_KH =
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>::Identity() -
      K * H;
  P = I_KH * P * I_KH.transpose() +
      K * Eigen::Matrix<double, 1, 1>::Constant(R) * K.transpose();

  // Ensure symmetry
  P = 0.5 * (P + P.transpose());
}

void EskfEstimator::InjectErrorAndReset(
    const Eigen::Matrix<double, kErrorStateSize, 1>& delta_x,
    Eigen::VectorXd& nominal_state,
    Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const {
  // Inject position error
  nominal_state.segment<3>(kPosIdx) += delta_x.segment<3>(kErrPosIdx);

  // Inject velocity error
  nominal_state.segment<3>(kVelIdx) += delta_x.segment<3>(kErrVelIdx);

  // Inject attitude error (multiplicative)
  // q_new = q ⊗ δq(δθ)
  Eigen::Vector3d delta_theta = delta_x.segment<3>(kErrAttIdx);
  Eigen::Vector4d delta_q = QuaternionFromAxisAngle(delta_theta);
  Eigen::Vector4d q = nominal_state.segment<4>(kQuatIdx);
  q = QuaternionMultiply(q, delta_q);
  q.normalize();
  nominal_state.segment<4>(kQuatIdx) = q;

  // Inject bias errors
  nominal_state.segment<3>(kAccelBiasIdx) +=
      delta_x.segment<3>(kErrAccelBiasIdx);
  nominal_state.segment<3>(kGyroBiasIdx) +=
      delta_x.segment<3>(kErrGyroBiasIdx);

  // Reset is implicit: error state is conceptually zeroed after injection
  // For the ESKF, we need to apply a reset matrix to P for proper covariance
  // For small errors, this is approximately identity (skip for simplicity)
}

// === State extraction helpers ===

Eigen::Vector3d EskfEstimator::ExtractPosition(
    const Eigen::VectorXd& state) const {
  return state.segment<3>(kPosIdx);
}

Eigen::Vector3d EskfEstimator::ExtractVelocity(
    const Eigen::VectorXd& state) const {
  return state.segment<3>(kVelIdx);
}

Eigen::Vector4d EskfEstimator::ExtractQuaternion(
    const Eigen::VectorXd& state) const {
  return state.segment<4>(kQuatIdx);
}

Eigen::Vector3d EskfEstimator::ExtractAccelBias(
    const Eigen::VectorXd& state) const {
  return state.segment<3>(kAccelBiasIdx);
}

Eigen::Vector3d EskfEstimator::ExtractGyroBias(
    const Eigen::VectorXd& state) const {
  return state.segment<3>(kGyroBiasIdx);
}

Eigen::Matrix<double, EskfEstimator::kErrorStateSize,
              EskfEstimator::kErrorStateSize>
EskfEstimator::ExtractCovariance(const Eigen::VectorXd& state) const {
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> P;
  for (int i = 0; i < kErrorStateSize; ++i) {
    for (int j = 0; j < kErrorStateSize; ++j) {
      P(i, j) = state[kNominalStateSize + i * kErrorStateSize + j];
    }
  }
  return P;
}

// === Quaternion utilities ===

Eigen::Vector4d EskfEstimator::QuaternionMultiply(
    const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) const {
  // Hamilton convention: q = [w, x, y, z]
  Eigen::Vector4d result;
  result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
  result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
  result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
  result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
  return result;
}

Eigen::Vector4d EskfEstimator::QuaternionFromAxisAngle(
    const Eigen::Vector3d& axis_angle) const {
  double angle = axis_angle.norm();
  Eigen::Vector4d q;
  if (angle < 1e-10) {
    // Small angle approximation
    q[0] = 1.0;
    q[1] = 0.5 * axis_angle[0];
    q[2] = 0.5 * axis_angle[1];
    q[3] = 0.5 * axis_angle[2];
  } else {
    Eigen::Vector3d axis = axis_angle / angle;
    double half_angle = 0.5 * angle;
    q[0] = std::cos(half_angle);
    double s = std::sin(half_angle);
    q[1] = s * axis[0];
    q[2] = s * axis[1];
    q[3] = s * axis[2];
  }
  return q;
}

Eigen::Matrix3d EskfEstimator::QuaternionToRotationMatrix(
    const Eigen::Vector4d& q) const {
  // q = [w, x, y, z]
  double w = q[0], x = q[1], y = q[2], z = q[3];

  Eigen::Matrix3d R;
  R(0, 0) = 1 - 2 * (y * y + z * z);
  R(0, 1) = 2 * (x * y - w * z);
  R(0, 2) = 2 * (x * z + w * y);
  R(1, 0) = 2 * (x * y + w * z);
  R(1, 1) = 1 - 2 * (x * x + z * z);
  R(1, 2) = 2 * (y * z - w * x);
  R(2, 0) = 2 * (x * z - w * y);
  R(2, 1) = 2 * (y * z + w * x);
  R(2, 2) = 1 - 2 * (x * x + y * y);

  return R;
}

Eigen::Matrix3d EskfEstimator::SkewSymmetric(const Eigen::Vector3d& v) const {
  Eigen::Matrix3d m;
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return m;
}

// === Output calculations ===

void EskfEstimator::CalcEstimatedPose(const Context<double>& context,
                                      BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  // [position(3), quaternion(4)]
  output->SetAtIndex(0, state[kPosIdx]);
  output->SetAtIndex(1, state[kPosIdx + 1]);
  output->SetAtIndex(2, state[kPosIdx + 2]);
  output->SetAtIndex(3, state[kQuatIdx]);      // w
  output->SetAtIndex(4, state[kQuatIdx + 1]);  // x
  output->SetAtIndex(5, state[kQuatIdx + 2]);  // y
  output->SetAtIndex(6, state[kQuatIdx + 3]);  // z
}

void EskfEstimator::CalcEstimatedVelocity(const Context<double>& context,
                                          BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  output->SetAtIndex(0, state[kVelIdx]);
  output->SetAtIndex(1, state[kVelIdx + 1]);
  output->SetAtIndex(2, state[kVelIdx + 2]);
}

void EskfEstimator::CalcEstimatedState(const Context<double>& context,
                                       BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  // [position(3), velocity(3)] - compatible with controller input
  output->SetAtIndex(0, state[kPosIdx]);
  output->SetAtIndex(1, state[kPosIdx + 1]);
  output->SetAtIndex(2, state[kPosIdx + 2]);
  output->SetAtIndex(3, state[kVelIdx]);
  output->SetAtIndex(4, state[kVelIdx + 1]);
  output->SetAtIndex(5, state[kVelIdx + 2]);
}

void EskfEstimator::CalcEstimatedBiases(const Context<double>& context,
                                        BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  // [accel_bias(3), gyro_bias(3)]
  for (int i = 0; i < 3; ++i) {
    output->SetAtIndex(i, state[kAccelBiasIdx + i]);
    output->SetAtIndex(3 + i, state[kGyroBiasIdx + i]);
  }
}

void EskfEstimator::CalcCovariance(const Context<double>& context,
                                   BasicVector<double>* output) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  // Extract diagonal of covariance matrix
  for (int i = 0; i < kErrorStateSize; ++i) {
    output->SetAtIndex(
        i, state[kNominalStateSize + i * kErrorStateSize + i]);
  }
}

void EskfEstimator::SetInitialState(
    Context<double>* context, const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity, const Eigen::Vector4d& quaternion,
    const Eigen::Vector3d& accel_bias,
    const Eigen::Vector3d& gyro_bias) const {
  auto& state = context->get_mutable_discrete_state(state_index_);

  state[kPosIdx] = position.x();
  state[kPosIdx + 1] = position.y();
  state[kPosIdx + 2] = position.z();

  state[kVelIdx] = velocity.x();
  state[kVelIdx + 1] = velocity.y();
  state[kVelIdx + 2] = velocity.z();

  // Normalize quaternion before setting
  Eigen::Vector4d q_normalized = quaternion.normalized();
  state[kQuatIdx] = q_normalized[0];
  state[kQuatIdx + 1] = q_normalized[1];
  state[kQuatIdx + 2] = q_normalized[2];
  state[kQuatIdx + 3] = q_normalized[3];

  state[kAccelBiasIdx] = accel_bias.x();
  state[kAccelBiasIdx + 1] = accel_bias.y();
  state[kAccelBiasIdx + 2] = accel_bias.z();

  state[kGyroBiasIdx] = gyro_bias.x();
  state[kGyroBiasIdx + 1] = gyro_bias.y();
  state[kGyroBiasIdx + 2] = gyro_bias.z();
}

Eigen::Vector4d EskfEstimator::GetQuaternion(
    const Context<double>& context) const {
  const auto& state = context.get_discrete_state(state_index_).value();
  return state.segment<4>(kQuatIdx);
}

Eigen::Matrix3d EskfEstimator::GetRotationMatrix(
    const Context<double>& context) const {
  return QuaternionToRotationMatrix(GetQuaternion(context));
}

}  // namespace quad_rope_lift
