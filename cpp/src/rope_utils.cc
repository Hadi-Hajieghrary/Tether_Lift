#include "rope_utils.h"

#include <cmath>
#include <algorithm>

namespace quad_rope_lift {

RopeParameters ComputeRopeParameters(
    int num_beads,
    double rope_rest_length,
    double rope_total_mass,
    double segment_stiffness,
    double segment_damping,
    bool bead_diameter_equals_spacing,
    double max_bead_radius) {

  RopeParameters params{};
  params.segment_stiffness = segment_stiffness;
  params.segment_damping = segment_damping;

  if (num_beads <= 0) {
    params.num_beads = 0;
    params.bead_mass = 0.0;
    params.bead_radius = 0.0;
    return params;
  }

  params.num_beads = num_beads;

  // Distribute mass equally among beads
  params.bead_mass = rope_total_mass / num_beads;

  // Segment rest length: rope divided into (num_beads + 1) segments
  // (quad -> bead_0 -> bead_1 -> ... -> bead_N-1 -> payload)
  const double segment_rest_length = rope_rest_length / (num_beads + 1);

  // Compute bead radius based on spacing
  if (bead_diameter_equals_spacing) {
    // Bead diameter ≈ segment length (beads nearly touching)
    // Use 0.49 factor to avoid initial contact/interpenetration
    params.bead_radius = 0.49 * segment_rest_length;
  } else {
    // Smaller beads for visual clarity
    params.bead_radius = 0.25 * segment_rest_length;
  }

  // Enforce maximum radius constraint
  params.bead_radius = std::min(params.bead_radius, max_bead_radius);

  return params;
}

std::vector<Eigen::Vector3d> GenerateSlackRopePositions(
    const Eigen::Vector3d& start_point_world,
    const Eigen::Vector3d& end_point_world,
    int num_beads,
    double rope_rest_length,
    double slack_ratio,
    const Eigen::Vector3d& lateral_direction,
    double max_lateral_amplitude) {

  std::vector<Eigen::Vector3d> bead_positions;

  if (num_beads <= 0) {
    return bead_positions;
  }

  const Eigen::Vector3d start = start_point_world;
  const Eigen::Vector3d end = end_point_world;

  // Target polyline length (slightly less than rest length to show slack)
  double target_arc_length = slack_ratio * rope_rest_length;
  const double chord_length = (end - start).norm();

  // If target is shorter than chord, just use straight line
  if (target_arc_length <= chord_length + 1e-12) {
    target_arc_length = chord_length;
  }

  // Normalize the lateral direction
  Eigen::Vector3d lateral = lateral_direction;
  const double lateral_norm = lateral.norm();
  if (lateral_norm < 1e-12) {
    lateral = Eigen::Vector3d::UnitX();
  } else {
    lateral /= lateral_norm;
  }

  // Lambda to compute polyline length for a given amplitude
  auto compute_polyline_length = [&](double amplitude) -> double {
    std::vector<Eigen::Vector3d> points;
    points.push_back(start);

    for (int i = 0; i < num_beads; ++i) {
      const double t = static_cast<double>(i + 1) / (num_beads + 1);
      const Eigen::Vector3d base_point = (1.0 - t) * start + t * end;
      const Eigen::Vector3d lateral_offset =
          amplitude * std::sin(M_PI * t) * lateral;
      points.push_back(base_point + lateral_offset);
    }
    points.push_back(end);

    double total_length = 0.0;
    for (size_t i = 0; i + 1 < points.size(); ++i) {
      total_length += (points[i + 1] - points[i]).norm();
    }
    return total_length;
  };

  // Binary search to find amplitude that gives target arc length
  double amplitude_low = 0.0;
  double amplitude_high = max_lateral_amplitude;
  double optimal_amplitude = 0.0;

  if (compute_polyline_length(amplitude_low) >= target_arc_length) {
    optimal_amplitude = 0.0;
  } else {
    // 40 iterations gives ~1e-12 precision
    for (int iter = 0; iter < 40; ++iter) {
      const double amplitude_mid = 0.5 * (amplitude_low + amplitude_high);
      if (compute_polyline_length(amplitude_mid) < target_arc_length) {
        amplitude_low = amplitude_mid;
      } else {
        amplitude_high = amplitude_mid;
      }
    }
    optimal_amplitude = amplitude_high;
  }

  // Generate bead positions with the computed amplitude
  bead_positions.reserve(num_beads);
  for (int i = 0; i < num_beads; ++i) {
    const double t = static_cast<double>(i + 1) / (num_beads + 1);
    const Eigen::Vector3d base_point = (1.0 - t) * start + t * end;
    const Eigen::Vector3d lateral_offset =
        optimal_amplitude * std::sin(M_PI * t) * lateral;
    bead_positions.push_back(base_point + lateral_offset);
  }

  return bead_positions;
}

}  // namespace quad_rope_lift
