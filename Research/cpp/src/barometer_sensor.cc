#include "barometer_sensor.h"

#include <cmath>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

// State indices within the discrete state vector
constexpr int kAltitudeMeasOffset = 0;
constexpr int kBiasOffset = 1;
constexpr int kCorrelatedNoiseOffset = 2;
constexpr int kValidOffset = 3;
constexpr int kTotalStateSize = 4;

BarometerSensor::BarometerSensor(const MultibodyPlant<double>& plant,
                                 const RigidBody<double>& body,
                                 const BarometerParams& params)
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

  // Discrete state: measurement + bias + correlated noise + valid flag
  Eigen::VectorXd initial_state(kTotalStateSize);
  initial_state.setZero();
  initial_state[kBiasOffset] = params_.initial_bias;
  initial_state[kValidOffset] = 1.0;  // Initially valid

  measurement_state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update at barometer sample rate
  DeclarePeriodicDiscreteUpdateEvent(params_.sample_period_sec, 0.0,
                                     &BarometerSensor::UpdateBaroMeasurement);

  // Output ports
  altitude_output_port_ =
      DeclareVectorOutputPort("altitude", BasicVector<double>(1),
                              &BarometerSensor::CalcAltitude)
          .get_index();

  valid_output_port_ =
      DeclareVectorOutputPort("baro_valid", BasicVector<double>(1),
                              &BarometerSensor::CalcBaroValid)
          .get_index();
}

EventStatus BarometerSensor::UpdateBaroMeasurement(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  // Get plant state
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  // Create temporary plant context to evaluate body pose
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Get true altitude (z-position in world frame)
  const auto& body = plant_.get_body(body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, body);
  const double true_altitude = pose_world.translation().z();

  // Get current state
  auto& state = discrete_state->get_mutable_vector(measurement_state_index_);
  double bias = state[kBiasOffset];
  double correlated_noise = state[kCorrelatedNoiseOffset];

  const double dt = params_.sample_period_sec;

  // === Update bias (slow random walk) ===
  // db/dt = drift_rate * w, where w is unit white noise
  bias += params_.bias_drift_rate * dt * noise_dist_(generator_);

  // === Update correlated noise (first-order Gauss-Markov) ===
  // dn/dt = -n/tau + eta, discrete: n[k+1] = alpha*n[k] + beta*sigma*w[k]
  const double alpha = std::exp(-dt / params_.correlation_time);
  const double beta = std::sqrt(1.0 - alpha * alpha);
  correlated_noise = alpha * correlated_noise +
                     beta * params_.correlated_noise_stddev *
                         noise_dist_(generator_);

  // === Compute white noise ===
  const double white_noise =
      params_.white_noise_stddev * noise_dist_(generator_);

  // === Total measurement ===
  double noisy_altitude =
      true_altitude + bias + correlated_noise + white_noise;

  // Apply quantization
  noisy_altitude = Quantize(noisy_altitude);

  // Store updated state
  state[kAltitudeMeasOffset] = noisy_altitude;
  state[kBiasOffset] = bias;
  state[kCorrelatedNoiseOffset] = correlated_noise;
  state[kValidOffset] = 1.0;

  return EventStatus::Succeeded();
}

void BarometerSensor::CalcAltitude(const Context<double>& context,
                                   BasicVector<double>* output) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  output->SetAtIndex(0, state[kAltitudeMeasOffset]);
}

void BarometerSensor::CalcBaroValid(const Context<double>& context,
                                    BasicVector<double>* output) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  output->SetAtIndex(0, state[kValidOffset]);
}

void BarometerSensor::InitializeState(Context<double>* context,
                                      double initial_altitude) const {
  auto& state = context->get_mutable_discrete_state(measurement_state_index_);
  // Initialize with true altitude (plus initial bias)
  state[kAltitudeMeasOffset] =
      Quantize(initial_altitude + params_.initial_bias);
  state[kCorrelatedNoiseOffset] = 0.0;  // Start with zero correlated noise
}

double BarometerSensor::GetBias(const Context<double>& context) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  return state[kBiasOffset];
}

double BarometerSensor::GetCorrelatedNoise(
    const Context<double>& context) const {
  const auto& state =
      context.get_discrete_state(measurement_state_index_).value();
  return state[kCorrelatedNoiseOffset];
}

double BarometerSensor::Quantize(double value) const {
  if (params_.resolution <= 0.0) {
    return value;  // No quantization
  }
  return std::round(value / params_.resolution) * params_.resolution;
}

// Factory functions for common barometer configurations

BarometerParams HighQualityBarometerParams() {
  BarometerParams params;
  // High-quality MEMS barometer (e.g., DPS310, MS5611)
  params.white_noise_stddev = 0.1;
  params.correlated_noise_stddev = 0.1;
  params.correlation_time = 3.0;
  params.bias_drift_rate = 0.001;  // 0.06 m/min
  params.sample_period_sec = 0.02;  // 50 Hz
  params.resolution = 0.06;
  return params;
}

BarometerParams StandardBarometerParams() {
  BarometerParams params;
  // Standard MEMS barometer (e.g., BMP280, LPS25H)
  params.white_noise_stddev = 0.3;
  params.correlated_noise_stddev = 0.2;
  params.correlation_time = 5.0;
  params.bias_drift_rate = 0.002;  // 0.12 m/min
  params.sample_period_sec = 0.04;  // 25 Hz
  params.resolution = 0.1;
  return params;
}

}  // namespace quad_rope_lift
