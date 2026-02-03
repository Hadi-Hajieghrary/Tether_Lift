#include "imu_sensor.h"

#include <cmath>

#include <drake/math/rotation_matrix.h>
#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::math::RotationMatrixd;
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

// State indices within the discrete state vector
constexpr int kGyroMeasOffset = 0;
constexpr int kAccelMeasOffset = 3;
constexpr int kGyroBiasOffset = 6;
constexpr int kAccelBiasOffset = 9;
constexpr int kValidOffset = 12;
constexpr int kTotalStateSize = 13;

ImuSensor::ImuSensor(const MultibodyPlant<double>& plant,
                     const RigidBody<double>& body,
                     const ImuParams& params)
    : plant_(plant),
      body_index_(body.index()),
      params_(params),
      generator_(params.random_seed) {
  // Input port: full plant state
  plant_state_port_ =
      DeclareVectorInputPort(
          "plant_state",
          BasicVector<double>(plant.num_positions() + plant.num_velocities()))
          .get_index();

  // Discrete state: measurements + biases + valid flag
  Eigen::VectorXd initial_state(kTotalStateSize);
  initial_state.setZero();

  // Initialize biases with specified initial values
  initial_state.segment<3>(kGyroBiasOffset) = params_.gyro_initial_bias;
  initial_state.segment<3>(kAccelBiasOffset) = params_.accel_initial_bias;
  initial_state[kValidOffset] = 1.0;  // Initially valid

  measurement_state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update at IMU sample rate
  DeclarePeriodicDiscreteUpdateEvent(params_.sample_period_sec, 0.0,
                                     &ImuSensor::UpdateImuMeasurement);

  // Output ports
  gyro_output_port_ =
      DeclareVectorOutputPort("gyro_measurement", BasicVector<double>(3),
                              &ImuSensor::CalcGyroMeasurement)
          .get_index();

  accel_output_port_ =
      DeclareVectorOutputPort("accel_measurement", BasicVector<double>(3),
                              &ImuSensor::CalcAccelMeasurement)
          .get_index();

  valid_output_port_ =
      DeclareVectorOutputPort("imu_valid", BasicVector<double>(1),
                              &ImuSensor::CalcImuValid)
          .get_index();
}

EventStatus ImuSensor::UpdateImuMeasurement(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  // Get plant state
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  // Create temporary plant context to evaluate body pose and velocity
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Get body pose and velocity
  const auto& body = plant_.get_body(body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, body);
  const auto& spatial_velocity =
      plant_.EvalBodySpatialVelocityInWorld(*plant_context, body);
  const auto& spatial_acceleration =
      plant_.EvalBodySpatialAccelerationInWorld(*plant_context, body);

  // Get rotation matrix (world to body)
  const RotationMatrixd& R_WB = pose_world.rotation();
  const Eigen::Matrix3d R_BW = R_WB.inverse().matrix();

  // Get current biases from state
  auto& state = discrete_state->get_mutable_vector(measurement_state_index_);
  Eigen::Vector3d gyro_bias = state.get_value().segment<3>(kGyroBiasOffset);
  Eigen::Vector3d accel_bias = state.get_value().segment<3>(kAccelBiasOffset);

  // === Gyroscope measurement ===
  // True angular velocity in body frame
  const Eigen::Vector3d omega_world = spatial_velocity.rotational();
  const Eigen::Vector3d omega_body = R_BW * omega_world;

  // Add white noise (scaled by sqrt(dt) for discrete-time)
  const double dt = params_.sample_period_sec;
  const double sqrt_dt = std::sqrt(dt);
  Eigen::Vector3d gyro_noise;
  for (int i = 0; i < 3; ++i) {
    gyro_noise[i] =
        params_.gyro_noise_density[i] * noise_dist_(generator_) / sqrt_dt;
  }

  // Gyro measurement = true + bias + noise
  const Eigen::Vector3d gyro_meas = omega_body + gyro_bias + gyro_noise;

  // === Accelerometer measurement ===
  // True linear acceleration in world frame
  const Eigen::Vector3d accel_world = spatial_acceleration.translational();

  // Specific force = acceleration - gravity (what accelerometer measures)
  const Eigen::Vector3d specific_force_world = accel_world - gravity_world_;

  // Transform to body frame
  const Eigen::Vector3d specific_force_body = R_BW * specific_force_world;

  // Add white noise
  Eigen::Vector3d accel_noise;
  for (int i = 0; i < 3; ++i) {
    accel_noise[i] =
        params_.accel_noise_density[i] * noise_dist_(generator_) / sqrt_dt;
  }

  // Accel measurement = true specific force + bias + noise
  const Eigen::Vector3d accel_meas = specific_force_body + accel_bias + accel_noise;

  // === Update biases (first-order Gauss-Markov process) ===
  // db/dt = -b/tau + eta, where eta is white noise with variance sigma^2
  // Discrete: b[k+1] = exp(-dt/tau) * b[k] + sqrt(1 - exp(-2*dt/tau)) * sigma * w[k]

  // Gyro bias update
  const double gyro_alpha = std::exp(-dt / params_.gyro_bias_time_constant);
  const double gyro_beta =
      std::sqrt(1.0 - gyro_alpha * gyro_alpha);  // Scale for noise
  for (int i = 0; i < 3; ++i) {
    gyro_bias[i] = gyro_alpha * gyro_bias[i] +
                   gyro_beta * params_.gyro_bias_instability[i] *
                       noise_dist_(generator_);
  }

  // Accel bias update
  const double accel_alpha = std::exp(-dt / params_.accel_bias_time_constant);
  const double accel_beta = std::sqrt(1.0 - accel_alpha * accel_alpha);
  for (int i = 0; i < 3; ++i) {
    accel_bias[i] = accel_alpha * accel_bias[i] +
                    accel_beta * params_.accel_bias_instability[i] *
                        noise_dist_(generator_);
  }

  // Store updated state
  state[kGyroMeasOffset + 0] = gyro_meas.x();
  state[kGyroMeasOffset + 1] = gyro_meas.y();
  state[kGyroMeasOffset + 2] = gyro_meas.z();

  state[kAccelMeasOffset + 0] = accel_meas.x();
  state[kAccelMeasOffset + 1] = accel_meas.y();
  state[kAccelMeasOffset + 2] = accel_meas.z();

  state[kGyroBiasOffset + 0] = gyro_bias.x();
  state[kGyroBiasOffset + 1] = gyro_bias.y();
  state[kGyroBiasOffset + 2] = gyro_bias.z();

  state[kAccelBiasOffset + 0] = accel_bias.x();
  state[kAccelBiasOffset + 1] = accel_bias.y();
  state[kAccelBiasOffset + 2] = accel_bias.z();

  state[kValidOffset] = 1.0;

  return EventStatus::Succeeded();
}

void ImuSensor::CalcGyroMeasurement(const Context<double>& context,
                                    BasicVector<double>* output) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  output->SetFromVector(state.segment<3>(kGyroMeasOffset));
}

void ImuSensor::CalcAccelMeasurement(const Context<double>& context,
                                     BasicVector<double>* output) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  output->SetFromVector(state.segment<3>(kAccelMeasOffset));
}

void ImuSensor::CalcImuValid(const Context<double>& context,
                             BasicVector<double>* output) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  output->SetAtIndex(0, state[kValidOffset]);
}

void ImuSensor::InitializeBiases(Context<double>* context,
                                 const Eigen::Vector3d& gyro_bias,
                                 const Eigen::Vector3d& accel_bias) const {
  auto& state =
      context->get_mutable_discrete_state(measurement_state_index_);
  state[kGyroBiasOffset + 0] = gyro_bias.x();
  state[kGyroBiasOffset + 1] = gyro_bias.y();
  state[kGyroBiasOffset + 2] = gyro_bias.z();
  state[kAccelBiasOffset + 0] = accel_bias.x();
  state[kAccelBiasOffset + 1] = accel_bias.y();
  state[kAccelBiasOffset + 2] = accel_bias.z();
}

Eigen::Vector3d ImuSensor::GetGyroBias(const Context<double>& context) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  return state.segment<3>(kGyroBiasOffset);
}

Eigen::Vector3d ImuSensor::GetAccelBias(const Context<double>& context) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  return state.segment<3>(kAccelBiasOffset);
}

// Factory functions for common IMU configurations

ImuParams ConsumerImuParams() {
  ImuParams params;
  // Typical consumer MEMS (e.g., MPU6050, BMI160)
  params.gyro_noise_density = Eigen::Vector3d::Constant(5e-4);  // 0.03 °/s/√Hz
  params.gyro_bias_instability =
      Eigen::Vector3d::Constant(3e-4);  // ~60 °/hr
  params.gyro_bias_time_constant = 1000.0;

  params.accel_noise_density =
      Eigen::Vector3d::Constant(4e-3);  // 400 μg/√Hz
  params.accel_bias_instability = Eigen::Vector3d::Constant(1e-3);  // 100 μg
  params.accel_bias_time_constant = 1000.0;

  params.sample_period_sec = 0.0025;  // 400 Hz
  return params;
}

ImuParams TacticalImuParams() {
  ImuParams params;
  // Tactical-grade MEMS (e.g., ADIS16488, VN-100)
  params.gyro_noise_density =
      Eigen::Vector3d::Constant(3e-5);  // 0.0016 °/s/√Hz
  params.gyro_bias_instability = Eigen::Vector3d::Constant(1e-5);  // ~2 °/hr
  params.gyro_bias_time_constant = 3600.0;

  params.accel_noise_density =
      Eigen::Vector3d::Constant(3e-4);  // 30 μg/√Hz
  params.accel_bias_instability = Eigen::Vector3d::Constant(5e-5);  // 5 μg
  params.accel_bias_time_constant = 3600.0;

  params.sample_period_sec = 0.002;  // 500 Hz
  return params;
}

}  // namespace quad_rope_lift
