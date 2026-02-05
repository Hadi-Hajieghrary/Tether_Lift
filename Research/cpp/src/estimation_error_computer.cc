#include "estimation_error_computer.h"

#include <cmath>

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;

EstimationErrorComputer::EstimationErrorComputer(
    const MultibodyPlant<double>& plant,
    const RigidBody<double>& body)
    : plant_(plant), body_index_(body.index()) {

  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  estimated_state_port_ = DeclareVectorInputPort(
      "estimated_state", BasicVector<double>(6))
      .get_index();

  // Output: [pos_err(3), vel_err(3), pos_norm, vel_norm] = 8 elements
  error_port_ = DeclareVectorOutputPort(
      "error", BasicVector<double>(8),
      &EstimationErrorComputer::CalcError)
      .get_index();
}

void EstimationErrorComputer::CalcError(
    const Context<double>& context,
    BasicVector<double>* output) const {

  // Get ground truth from plant
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  const auto& body = plant_.get_body(body_index_);
  const auto& pose = plant_.EvalBodyPoseInWorld(*plant_context, body);
  const auto& velocity = plant_.EvalBodySpatialVelocityInWorld(*plant_context, body);

  const Eigen::Vector3d true_pos = pose.translation();
  const Eigen::Vector3d true_vel = velocity.translational();

  // Get estimated state
  const auto& estimated = get_input_port(estimated_state_port_).Eval(context);
  const Eigen::Vector3d est_pos(estimated[0], estimated[1], estimated[2]);
  const Eigen::Vector3d est_vel(estimated[3], estimated[4], estimated[5]);

  // Compute errors
  const Eigen::Vector3d pos_err = est_pos - true_pos;
  const Eigen::Vector3d vel_err = est_vel - true_vel;

  // Output
  output->SetAtIndex(0, pos_err.x());
  output->SetAtIndex(1, pos_err.y());
  output->SetAtIndex(2, pos_err.z());
  output->SetAtIndex(3, vel_err.x());
  output->SetAtIndex(4, vel_err.y());
  output->SetAtIndex(5, vel_err.z());
  output->SetAtIndex(6, pos_err.norm());
  output->SetAtIndex(7, vel_err.norm());
}

}  // namespace quad_rope_lift
