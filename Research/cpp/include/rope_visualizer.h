#pragma once

#include <Eigen/Core>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/rgba.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Draws the rope as a continuous polyline in Meshcat.
///
/// This system reads the plant state and draws a line connecting all the
/// attachment points (quadcopter -> beads -> payload) for visual clarity.
class RopeVisualizer final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RopeVisualizer);

  /// Constructs the rope visualizer.
  ///
  /// @param plant The MultibodyPlant containing all bodies.
  /// @param body_attachment_points Vector of (body, point_in_body_frame) pairs
  ///        defining the rope path from quadcopter to payload.
  /// @param meshcat The Meshcat instance for visualization.
  /// @param meshcat_path Path in Meshcat scene tree for the line.
  /// @param line_width Width of the polyline in pixels.
  /// @param line_color RGBA color of the line.
  /// @param update_period Time between visualization updates [s].
  RopeVisualizer(
      const drake::multibody::MultibodyPlant<double>& plant,
      const std::vector<std::pair<const drake::multibody::RigidBody<double>*,
                                  Eigen::Vector3d>>& body_attachment_points,
      std::shared_ptr<drake::geometry::Meshcat> meshcat,
      const std::string& meshcat_path = "rope_visualization",
      double line_width = 3.0,
      const drake::geometry::Rgba& line_color = drake::geometry::Rgba(0.6, 0.6, 0.6, 1.0),
      double update_period = 1.0 / 60.0);

  /// Returns the input port for plant state.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

 private:
  // Periodic publish event handler
  drake::systems::EventStatus UpdateVisualization(
      const drake::systems::Context<double>& context) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Bodies and attachment points
  std::vector<drake::multibody::BodyIndex> body_indices_;
  std::vector<Eigen::Vector3d> attachment_points_;

  // Meshcat reference
  std::shared_ptr<drake::geometry::Meshcat> meshcat_;
  std::string meshcat_path_;
  double line_width_;
  drake::geometry::Rgba line_color_;

  // Port index
  int plant_state_port_{-1};
};

}  // namespace quad_rope_lift
