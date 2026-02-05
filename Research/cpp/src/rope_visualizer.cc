#include "rope_visualizer.h"

namespace quad_rope_lift {

using drake::geometry::Meshcat;
using drake::geometry::Rgba;
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::EventStatus;

RopeVisualizer::RopeVisualizer(
    const MultibodyPlant<double>& plant,
    const std::vector<std::pair<const RigidBody<double>*, Eigen::Vector3d>>&
        body_attachment_points,
    std::shared_ptr<Meshcat> meshcat,
    const std::string& meshcat_path,
    double line_width,
    const Rgba& line_color,
    double update_period)
    : plant_(plant),
      meshcat_(std::move(meshcat)),
      meshcat_path_(meshcat_path),
      line_width_(line_width),
      line_color_(line_color) {

  // Cache body indices and attachment points
  body_indices_.reserve(body_attachment_points.size());
  attachment_points_.reserve(body_attachment_points.size());

  for (const auto& [body, point] : body_attachment_points) {
    body_indices_.push_back(body->index());
    attachment_points_.push_back(point);
  }

  // Declare input port for plant state
  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  // Declare periodic publish event
  DeclarePeriodicPublishEvent(
      update_period, 0.0,
      &RopeVisualizer::UpdateVisualization);
}

EventStatus RopeVisualizer::UpdateVisualization(
    const Context<double>& context) const {

  // Get plant state input
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  // Create a plant context and set state
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  const int num_points = static_cast<int>(body_indices_.size());

  // Compute world positions (3 x N matrix, column-major for Meshcat)
  Eigen::Matrix3Xd points_world(3, num_points);

  for (int i = 0; i < num_points; ++i) {
    const auto& body = plant_.get_body(body_indices_[i]);
    const auto& X_WB = plant_.EvalBodyPoseInWorld(*plant_context, body);
    points_world.col(i) = X_WB * attachment_points_[i];
  }

  // Draw the polyline
  meshcat_->SetLine(meshcat_path_, points_world, line_width_, line_color_);

  return EventStatus::Succeeded();
}

}  // namespace quad_rope_lift
