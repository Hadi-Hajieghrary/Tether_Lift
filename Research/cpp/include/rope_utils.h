#pragma once

#include <Eigen/Core>
#include <drake/geometry/meshcat.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the rope bead chain.
struct RopeParameters {
  int num_beads;
  double bead_mass;
  double bead_radius;
  double segment_stiffness;
  double segment_damping;
};

/// Compute physical parameters for each bead and segment of the rope chain.
///
/// The rope is modeled as a chain of spherical beads connected by spring-damper
/// segments. This function distributes the total rope mass among the beads and
/// computes appropriate radii for collision/visualization.
///
/// @param num_beads Number of beads in the chain (N).
/// @param rope_rest_length Total unstretched length of the rope [m].
/// @param rope_total_mass Total mass of the rope [kg].
/// @param segment_stiffness Spring constant for each segment [N/m].
/// @param segment_damping Damping coefficient for each segment [N·s/m].
/// @param bead_diameter_equals_spacing If true, bead diameter ≈ segment length.
/// @param max_bead_radius Maximum allowed bead radius [m].
/// @return RopeParameters struct with computed values.
RopeParameters ComputeRopeParameters(
    int num_beads,
    double rope_rest_length,
    double rope_total_mass,
    double segment_stiffness,
    double segment_damping,
    bool bead_diameter_equals_spacing = true,
    double max_bead_radius = 100.0);

/// Generate initial bead positions for a visually slack rope.
///
/// The rope hangs in a sine-wave-like curve between the start and end points.
/// This creates a realistic initial configuration where the rope is not fully
/// stretched, allowing for natural dynamics when the quadcopter takes off.
///
/// @param start_point_world Rope attachment point on the quadcopter (world frame).
/// @param end_point_world Rope attachment point on the payload (world frame).
/// @param num_beads Number of beads in the rope chain.
/// @param rope_rest_length The unstretched length of the rope.
/// @param slack_ratio Target arc length as a fraction of rest length (< 1 = slack).
/// @param lateral_direction Direction of the lateral bulge (perpendicular to rope).
/// @param max_lateral_amplitude Maximum lateral displacement for the curve.
/// @return Vector of 3D positions for each bead in world coordinates.
std::vector<Eigen::Vector3d> GenerateSlackRopePositions(
    const Eigen::Vector3d& start_point_world,
    const Eigen::Vector3d& end_point_world,
    int num_beads,
    double rope_rest_length,
    double slack_ratio = 0.85,
    const Eigen::Vector3d& lateral_direction = Eigen::Vector3d::UnitX(),
    double max_lateral_amplitude = 1.5);

}  // namespace quad_rope_lift
