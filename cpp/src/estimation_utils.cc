#include "estimation_utils.h"

#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;

// =============================================================================
// AttachmentPositionExtractor
// =============================================================================

AttachmentPositionExtractor::AttachmentPositionExtractor(
    const MultibodyPlant<double>& plant,
    const std::vector<const RigidBody<double>*>& bodies,
    const std::vector<Eigen::Vector3d>& offsets)
    : plant_(plant), offsets_(offsets) {

  DRAKE_DEMAND(bodies.size() == offsets.size());

  for (const auto* body : bodies) {
    body_indices_.push_back(body->index());
  }

  const int num_bodies = static_cast<int>(bodies.size());

  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  positions_port_ = DeclareVectorOutputPort(
      "positions", BasicVector<double>(3 * num_bodies),
      &AttachmentPositionExtractor::CalcPositions)
      .get_index();
}

void AttachmentPositionExtractor::CalcPositions(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  for (size_t i = 0; i < body_indices_.size(); ++i) {
    const auto& body = plant_.get_body(body_indices_[i]);
    const auto& pose = plant_.EvalBodyPoseInWorld(*plant_context, body);

    // Transform local offset to world frame
    const Eigen::Vector3d world_pos = pose * offsets_[i];

    output->SetAtIndex(3 * i + 0, world_pos.x());
    output->SetAtIndex(3 * i + 1, world_pos.y());
    output->SetAtIndex(3 * i + 2, world_pos.z());
  }
}

// =============================================================================
// CableLengthSource
// =============================================================================

CableLengthSource::CableLengthSource(const std::vector<double>& lengths)
    : lengths_(lengths) {

  DeclareVectorOutputPort(
      "cable_lengths", BasicVector<double>(static_cast<int>(lengths.size())),
      &CableLengthSource::CalcLengths);
}

void CableLengthSource::CalcLengths(const Context<double>& /*context*/,
                                    BasicVector<double>* output) const {
  for (size_t i = 0; i < lengths_.size(); ++i) {
    output->SetAtIndex(static_cast<int>(i), lengths_[i]);
  }
}

// =============================================================================
// TensionAggregator
// =============================================================================

TensionAggregator::TensionAggregator(int num_cables)
    : num_cables_(num_cables) {

  // Input ports: one per cable, each with [T, fx, fy, fz]
  for (int i = 0; i < num_cables; ++i) {
    DeclareVectorInputPort("tension_" + std::to_string(i), BasicVector<double>(4));
  }

  DeclareVectorOutputPort(
      "tensions", BasicVector<double>(num_cables),
      &TensionAggregator::CalcTensions);
}

void TensionAggregator::CalcTensions(const Context<double>& context,
                                     BasicVector<double>* output) const {
  for (int i = 0; i < num_cables_; ++i) {
    const auto& tension_vec = get_input_port(i).Eval(context);
    output->SetAtIndex(i, tension_vec[0]);  // First element is magnitude
  }
}

}  // namespace quad_rope_lift
