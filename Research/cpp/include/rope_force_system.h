#pragma once

#include <Eigen/Core>
#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Computes spring-damper forces for the rope bead chain.
///
/// The rope is modeled as a series of tension-only spring-damper segments:
/// - When stretched (distance > rest length): exerts restoring force
/// - When slack (distance <= rest length): exerts zero force
///
/// This system outputs:
/// 1. External spatial forces to apply to each body (quad, beads, payload)
/// 2. The tension magnitude and force vector of the top segment (for the controller)
class RopeForceSystem final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RopeForceSystem);

  /// Constructs the rope force system.
  ///
  /// @param plant The MultibodyPlant containing all bodies.
  /// @param quadcopter_body The quadcopter RigidBody.
  /// @param payload_body The payload RigidBody.
  /// @param bead_bodies Vector of bead RigidBodies (in order from quad to payload).
  /// @param quadcopter_attachment_point Attachment point in quad body frame.
  /// @param payload_attachment_point Attachment point in payload body frame.
  /// @param rope_rest_length Total unstretched length of the rope [m].
  /// @param segment_stiffness Spring constant for each segment [N/m].
  /// @param segment_damping Damping coefficient for each segment [N·s/m].
  /// @param min_distance_threshold Minimum distance to avoid divide-by-zero.
  RopeForceSystem(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::multibody::RigidBody<double>& quadcopter_body,
      const drake::multibody::RigidBody<double>& payload_body,
      const std::vector<const drake::multibody::RigidBody<double>*>& bead_bodies,
      const Eigen::Vector3d& quadcopter_attachment_point,
      const Eigen::Vector3d& payload_attachment_point,
      double rope_rest_length,
      double segment_stiffness,
      double segment_damping,
      double min_distance_threshold = 1e-9);

  /// Returns the input port for plant state.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Returns the output port for external forces.
  const drake::systems::OutputPort<double>& get_forces_output_port() const {
    return get_output_port(forces_port_);
  }

  /// Returns the output port for top segment tension [tension, fx, fy, fz].
  const drake::systems::OutputPort<double>& get_tension_output_port() const {
    return get_output_port(tension_port_);
  }

 private:
  // Output computation methods
  void CalcRopeForces(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  void CalcTopSegmentTension(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Body references (indices for faster lookup)
  std::vector<drake::multibody::BodyIndex> body_indices_;

  // Attachment points in body frames
  std::vector<Eigen::Vector3d> attachment_points_;

  // Rope parameters
  int num_segments_;
  double segment_rest_length_;
  double segment_stiffness_;
  double segment_damping_;
  double min_distance_;

  // Port indices
  int plant_state_port_{-1};
  int forces_port_{-1};
  int tension_port_{-1};
};

}  // namespace quad_rope_lift
