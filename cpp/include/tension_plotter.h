#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/rgba.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Real-time tension plotter for Meshcat visualization.
///
/// Displays a scrolling line plot of rope tensions over time in the
/// Meshcat browser interface.
class TensionPlotter final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TensionPlotter);

  /// Constructs the tension plotter.
  ///
  /// @param meshcat Shared pointer to the Meshcat instance.
  /// @param num_ropes Number of rope tension signals to plot.
  /// @param colors Colors for each rope's plot line.
  /// @param time_window Duration of history to display [s].
  /// @param max_tension Maximum expected tension for y-axis scaling [N].
  /// @param update_period How often to update the plot [s].
  TensionPlotter(
      std::shared_ptr<drake::geometry::Meshcat> meshcat,
      int num_ropes,
      const std::vector<drake::geometry::Rgba>& colors,
      double time_window = 10.0,
      double max_tension = 50.0,
      double update_period = 0.05);

  /// Returns the input port for rope tension at index i.
  /// Each port expects a 4-element vector [tension, fx, fy, fz].
  const drake::systems::InputPort<double>& get_tension_input_port(int i) const {
    return get_input_port(tension_ports_.at(i));
  }

 private:
  // Periodic update event
  drake::systems::EventStatus UpdatePlots(
      const drake::systems::Context<double>& context) const;

  // Meshcat instance
  std::shared_ptr<drake::geometry::Meshcat> meshcat_;

  // Configuration
  int num_ropes_;
  std::vector<drake::geometry::Rgba> colors_;
  double time_window_;
  double max_tension_;

  // Port indices
  std::vector<int> tension_ports_;

  // Mutable history buffers (updated during output computation)
  mutable std::vector<std::deque<double>> tension_history_;
  mutable std::deque<double> time_history_;
  mutable double last_update_time_;
};

}  // namespace quad_rope_lift
