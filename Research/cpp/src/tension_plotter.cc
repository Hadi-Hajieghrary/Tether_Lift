#include "tension_plotter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace quad_rope_lift {

using drake::geometry::Meshcat;
using drake::geometry::Rgba;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::EventStatus;

TensionPlotter::TensionPlotter(
    std::shared_ptr<Meshcat> meshcat,
    int num_ropes,
    const std::vector<Rgba>& colors,
    double time_window,
    double max_tension,
    double update_period)
    : meshcat_(std::move(meshcat)),
      num_ropes_(num_ropes),
      colors_(colors),
      time_window_(time_window),
      max_tension_(max_tension),
      tension_history_(num_ropes),
      last_update_time_(-1.0) {

  // Ensure we have enough colors
  while (colors_.size() < static_cast<size_t>(num_ropes_)) {
    colors_.push_back(Rgba(0.5, 0.5, 0.5, 1.0));
  }

  // Declare input ports for each rope's tension
  tension_ports_.reserve(num_ropes_);
  for (int i = 0; i < num_ropes_; ++i) {
    const std::string port_name = "tension_" + std::to_string(i);
    tension_ports_.push_back(
        DeclareVectorInputPort(port_name, BasicVector<double>(4)).get_index());
  }

  // Declare periodic publish event
  DeclarePeriodicPublishEvent(
      update_period, 0.0,
      &TensionPlotter::UpdatePlots);
}

EventStatus TensionPlotter::UpdatePlots(
    const Context<double>& context) const {

  const double t = context.get_time();

  // Read current tensions
  std::vector<double> current_tensions(num_ropes_);
  for (int i = 0; i < num_ropes_; ++i) {
    const auto& tension_data = get_input_port(tension_ports_[i]).Eval(context);
    current_tensions[i] = tension_data[0];  // First element is tension magnitude
  }

  // Add to history
  time_history_.push_back(t);
  for (int i = 0; i < num_ropes_; ++i) {
    tension_history_[i].push_back(current_tensions[i]);
  }

  // Remove old data outside the time window
  while (!time_history_.empty() && (t - time_history_.front()) > time_window_) {
    time_history_.pop_front();
    for (int i = 0; i < num_ropes_; ++i) {
      tension_history_[i].pop_front();
    }
  }

  const int num_points = static_cast<int>(time_history_.size());
  if (num_points < 2) {
    return EventStatus::Succeeded();
  }

  // Compute time range for x-axis
  const double t_min = time_history_.front();
  const double t_max = time_history_.back();
  const double t_range = std::max(t_max - t_min, 0.1);

  // Create 3D line plot for each rope's tension history
  // Position the plot at a fixed location in the scene (e.g., x=-3 to -1, z=4)
  const double plot_x_min = -3.5;
  const double plot_x_max = -1.0;
  const double plot_z_min = 3.5;
  const double plot_z_max = 5.0;
  const double plot_y = -2.0;  // Fixed y position

  const double x_scale = plot_x_max - plot_x_min;
  const double z_scale = plot_z_max - plot_z_min;

  // Draw each rope's tension history as a 3D polyline
  for (int i = 0; i < num_ropes_; ++i) {
    // Build the line data: 3 x N matrix
    Eigen::Matrix3Xd line_data(3, num_points);

    for (int j = 0; j < num_points; ++j) {
      // X: time normalized to plot range
      const double normalized_time = (time_history_[j] - t_min) / t_range;
      line_data(0, j) = plot_x_min + normalized_time * x_scale;

      // Y: offset per rope for visibility
      line_data(1, j) = plot_y + i * 0.05;

      // Z: normalized tension
      const double normalized_tension = std::clamp(
          tension_history_[i][j] / max_tension_, 0.0, 1.0);
      line_data(2, j) = plot_z_min + normalized_tension * z_scale;
    }

    const std::string path = "/TensionPlot/rope_" + std::to_string(i);
    meshcat_->SetLine(path, line_data, 3.0, colors_[i]);
  }

  // Draw axes/frame for the plot
  // Horizontal axis (time)
  Eigen::Matrix3Xd x_axis(3, 2);
  x_axis << plot_x_min, plot_x_max,
            plot_y, plot_y,
            plot_z_min, plot_z_min;
  meshcat_->SetLine("/TensionPlot/x_axis", x_axis, 1.0, Rgba(0.5, 0.5, 0.5, 0.8));

  // Vertical axis (tension)
  Eigen::Matrix3Xd z_axis(3, 2);
  z_axis << plot_x_min, plot_x_min,
            plot_y, plot_y,
            plot_z_min, plot_z_max;
  meshcat_->SetLine("/TensionPlot/z_axis", z_axis, 1.0, Rgba(0.5, 0.5, 0.5, 0.8));

  // Reference line at expected tension
  const double expected_tension = max_tension_ / 2.5;  // We set max_tension = 2.5 * expected
  const double ref_z = plot_z_min + (expected_tension / max_tension_) * z_scale;
  Eigen::Matrix3Xd ref_line(3, 2);
  ref_line << plot_x_min, plot_x_max,
              plot_y - 0.02, plot_y - 0.02,
              ref_z, ref_z;
  meshcat_->SetLine("/TensionPlot/reference", ref_line, 1.5, Rgba(0.8, 0.8, 0.8, 0.5));

  // Add current tension values as vertical bars at the right edge
  for (int i = 0; i < num_ropes_; ++i) {
    const double bar_x = plot_x_max + 0.1 + i * 0.15;
    const double bar_z = plot_z_min +
        std::clamp(current_tensions[i] / max_tension_, 0.0, 1.0) * z_scale;

    Eigen::Matrix3Xd bar(3, 2);
    bar << bar_x, bar_x,
           plot_y, plot_y,
           plot_z_min, bar_z;

    const std::string bar_path = "/TensionPlot/bar_" + std::to_string(i);
    meshcat_->SetLine(bar_path, bar, 8.0, colors_[i]);
  }

  return EventStatus::Succeeded();
}

}  // namespace quad_rope_lift
