#pragma once

#include <Eigen/Core>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/rgba.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>
#include <deque>
#include <functional>
#include <vector>

namespace tether_lift {

/**
 * @brief Visualizes trajectories in Meshcat: reference path and actual trails.
 *
 * This system provides:
 * 1. Static reference trajectory visualization (drawn once at start)
 * 2. Dynamic trail visualization for load and drones (updated as simulation runs)
 *
 * The reference trajectory is drawn as a dashed or solid line showing where
 * the load should go. The actual trails show where bodies have been.
 */
class TrajectoryVisualizer final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TrajectoryVisualizer);

  struct TrailConfig {
    std::string name;                    // Body name for labeling
    drake::multibody::BodyIndex body_index;
    Eigen::Vector3d local_point;         // Point in body frame to track
    drake::geometry::Rgba color;
    double line_width = 2.0;
    int max_trail_points = 500;          // Max points to keep in trail
  };

  struct Params {
    // Reference trajectory visualization
    bool show_reference_trajectory;
    drake::geometry::Rgba reference_color;
    double reference_line_width;
    int reference_sample_points;
    double trajectory_duration;

    // Trail visualization
    bool show_trails;
    double trail_update_period;
    double visualization_period;

    // Load trail config
    drake::geometry::Rgba load_trail_color;
    double load_trail_width;
    int load_max_trail_points;

    // Drone trail colors (will cycle if more drones than colors)
    std::vector<drake::geometry::Rgba> drone_trail_colors;
    double drone_trail_width;
    int drone_max_trail_points;

    Params()
        : show_reference_trajectory(true)
        , reference_color(0.0, 1.0, 0.0, 0.8)  // Green
        , reference_line_width(4.0)
        , reference_sample_points(200)
        , trajectory_duration(30.0)
        , show_trails(true)
        , trail_update_period(0.05)
        , visualization_period(1.0 / 30.0)
        , load_trail_color(1.0, 0.5, 0.0, 0.9)  // Orange
        , load_trail_width(3.0)
        , load_max_trail_points(1000)
        , drone_trail_colors{
            drake::geometry::Rgba(0.0, 0.5, 1.0, 0.7),   // Blue
            drake::geometry::Rgba(1.0, 0.0, 0.5, 0.7),   // Magenta
            drake::geometry::Rgba(0.5, 0.0, 1.0, 0.7),   // Purple
            drake::geometry::Rgba(0.0, 1.0, 1.0, 0.7),   // Cyan
            drake::geometry::Rgba(1.0, 1.0, 0.0, 0.7),   // Yellow
            drake::geometry::Rgba(1.0, 0.0, 0.0, 0.7)}   // Red
        , drone_trail_width(2.0)
        , drone_max_trail_points(500) {}
  };

  /**
   * @brief Construct trajectory visualizer.
   *
   * @param plant MultibodyPlant for computing body poses
   * @param meshcat Meshcat instance
   * @param load_body_index Index of the load body
   * @param drone_body_indices Indices of drone bodies
   * @param params Visualization parameters
   */
  TrajectoryVisualizer(
      const drake::multibody::MultibodyPlant<double>& plant,
      std::shared_ptr<drake::geometry::Meshcat> meshcat,
      drake::multibody::BodyIndex load_body_index,
      const std::vector<drake::multibody::BodyIndex>& drone_body_indices,
      const Params& params = Params());

  // === Input ports ===

  /// Plant state input (positions + velocities)
  const drake::systems::InputPort<double>& get_plant_state_input() const {
    return get_input_port(plant_state_port_);
  }

  /// Reference trajectory input: [p_des(3), v_des(3), a_des(3)] = 9D
  /// This is optional - if not connected, no reference trajectory shown
  const drake::systems::InputPort<double>& get_reference_trajectory_input() const {
    return get_input_port(reference_trajectory_port_);
  }

  /**
   * @brief Draw the full reference trajectory at the start of simulation.
   *
   * Call this after building the diagram but before running the simulation.
   * It samples the trajectory generator at multiple time points.
   *
   * @param waypoints Vector of 3D positions defining the reference path
   */
  void DrawReferenceTrajectory(const std::vector<Eigen::Vector3d>& waypoints);

  /**
   * @brief Draw reference trajectory from time-parameterized function.
   *
   * @param trajectory_func Function that returns position for given time
   * @param start_time Start time [s]
   * @param end_time End time [s]
   * @param num_samples Number of sample points
   */
  void DrawReferenceTrajectoryFromFunction(
      std::function<Eigen::Vector3d(double)> trajectory_func,
      double start_time,
      double end_time,
      int num_samples = 200);

  /**
   * @brief Clear all trails (but keep reference trajectory).
   */
  void ClearTrails();

  /**
   * @brief Clear everything including reference trajectory.
   */
  void ClearAll();

  const Params& params() const { return params_; }

 private:
  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  drake::systems::EventStatus UpdateAndDrawTrails(
      const drake::systems::Context<double>& context) const;

  // Helper to get world position of a body point
  Eigen::Vector3d GetBodyPointInWorld(
      const drake::systems::Context<double>& plant_context,
      drake::multibody::BodyIndex body_index,
      const Eigen::Vector3d& local_point) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Meshcat reference
  std::shared_ptr<drake::geometry::Meshcat> meshcat_;

  // Body tracking
  drake::multibody::BodyIndex load_body_index_;
  std::vector<drake::multibody::BodyIndex> drone_body_indices_;

  // Parameters
  Params params_;

  // Port indices
  int plant_state_port_{};
  int reference_trajectory_port_{};

  // Meshcat paths
  std::string reference_path_ = "trajectories/reference";
  std::string load_trail_path_ = "trajectories/load_trail";
  std::string drone_trail_base_path_ = "trajectories/drone_";

  // Trail storage (mutable for const publish methods)
  mutable std::deque<Eigen::Vector3d> load_trail_;
  mutable std::vector<std::deque<Eigen::Vector3d>> drone_trails_;
  mutable double last_trail_update_time_ = -1e9;
};

}  // namespace tether_lift
