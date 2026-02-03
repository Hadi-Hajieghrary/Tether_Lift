#include "gps_sensor.h"

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

GpsSensor::GpsSensor(const MultibodyPlant<double>& plant,
                     const RigidBody<double>& body,
                     const GpsParams& params)
    : plant_(plant),
      body_index_(body.index()),
      noise_stddev_(params.position_noise_stddev),
      dropout_probability_(params.dropout_probability),
      generator_(params.random_seed) {

  // Input port: full plant state
  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  // Discrete state for GPS measurement (x, y, z) and valid flag
  position_state_index_ = DeclareDiscreteState(3);  // [x, y, z]
  valid_state_index_ = DeclareDiscreteState(1);     // [valid]

  // Periodic update at GPS sample rate
  DeclarePeriodicDiscreteUpdateEvent(
      params.sample_period_sec, 0.0,
      &GpsSensor::UpdateGpsMeasurement);

  // Output ports
  position_output_port_ = DeclareVectorOutputPort(
      "gps_position", BasicVector<double>(3),
      &GpsSensor::CalcGpsPosition)
      .get_index();

  valid_output_port_ = DeclareVectorOutputPort(
      "gps_valid", BasicVector<double>(1),
      &GpsSensor::CalcGpsValid)
      .get_index();
}

EventStatus GpsSensor::UpdateGpsMeasurement(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {

  // Get plant state
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  // Create temporary plant context to evaluate body pose
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Get true body position
  const auto& body = plant_.get_body(body_index_);
  const auto& pose = plant_.EvalBodyPoseInWorld(*plant_context, body);
  const Eigen::Vector3d true_position = pose.translation();

  // Check for dropout
  const bool dropout = dropout_dist_(generator_) < dropout_probability_;

  if (dropout) {
    // Keep previous measurement, mark as invalid
    discrete_state->get_mutable_vector(valid_state_index_).SetAtIndex(0, 0.0);
  } else {
    // Add noise to true position
    Eigen::Vector3d noisy_position;
    for (int i = 0; i < 3; ++i) {
      noisy_position[i] = true_position[i] +
          noise_stddev_[i] * noise_dist_(generator_);
    }

    // Store in discrete state
    auto& pos_state = discrete_state->get_mutable_vector(position_state_index_);
    pos_state.SetAtIndex(0, noisy_position.x());
    pos_state.SetAtIndex(1, noisy_position.y());
    pos_state.SetAtIndex(2, noisy_position.z());

    discrete_state->get_mutable_vector(valid_state_index_).SetAtIndex(0, 1.0);
  }

  return EventStatus::Succeeded();
}

void GpsSensor::CalcGpsPosition(const Context<double>& context,
                                BasicVector<double>* output) const {
  const auto& pos_state = context.get_discrete_state(position_state_index_);
  output->SetAtIndex(0, pos_state[0]);
  output->SetAtIndex(1, pos_state[1]);
  output->SetAtIndex(2, pos_state[2]);
}

void GpsSensor::CalcGpsValid(const Context<double>& context,
                             BasicVector<double>* output) const {
  const auto& valid_state = context.get_discrete_state(valid_state_index_);
  output->SetAtIndex(0, valid_state[0]);
}

void GpsSensor::InitializeGpsState(Context<double>* context,
                                   const Eigen::Vector3d& initial_position) const {
  // Add initial noise (but don't mark as invalid)
  Eigen::Vector3d noisy_position;
  for (int i = 0; i < 3; ++i) {
    noisy_position[i] = initial_position[i] +
        noise_stddev_[i] * noise_dist_(generator_);
  }

  // Store in discrete state
  auto& pos_state = context->get_mutable_discrete_state(position_state_index_);
  pos_state[0] = noisy_position.x();
  pos_state[1] = noisy_position.y();
  pos_state[2] = noisy_position.z();

  // Mark as valid
  context->get_mutable_discrete_state(valid_state_index_)[0] = 1.0;
}

}  // namespace quad_rope_lift
