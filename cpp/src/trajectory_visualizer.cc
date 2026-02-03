#include "trajectory_visualizer.h"

#include <drake/math/rigid_transform.h>
#include <fmt/format.h>

namespace tether_lift {

using drake::geometry::Meshcat;
using drake::geometry::Rgba;
using drake::math::RigidTransformd;
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

TrajectoryVisualizer::TrajectoryVisualizer(
    const MultibodyPlant<double>& plant,
    std::shared_ptr<Meshcat> meshcat,
    BodyIndex load_body_index,
    const std::vector<BodyIndex>& drone_body_indices,
    const Params& params)
    : plant_(plant),
      meshcat_(std::move(meshcat)),
      load_body_index_(load_body_index),
      drone_body_indices_(drone_body_indices),
      params_(params) {
  // Create input ports
  plant_state_port_ = this->DeclareVectorInputPort(
      "plant_state", plant_.num_positions() + plant_.num_velocities())
      .get_index();

  reference_trajectory_port_ = this->DeclareVectorInputPort(
      "reference_trajectory", 9)  // [p_des(3), v_des(3), a_des(3)]
      .get_index();

  // Initialize drone trails vector
  drone_trails_.resize(drone_body_indices_.size());

  // Declare periodic publish event for drawing trails
  // We use publish events with mutable storage since we don't need
  // to persist trail data across context operations
  if (params_.show_trails) {
    this->DeclarePeriodicPublishEvent(
        params_.visualization_period,
        0.0,
        &TrajectoryVisualizer::UpdateAndDrawTrails);
  }
}

void TrajectoryVisualizer::SetDefaultState(
    const Context<double>& /* context */,
    drake::systems::State<double>* /* state */) const {
  // Clear trails when state is reset
  load_trail_.clear();
  for (auto& trail : drone_trails_) {
    trail.clear();
  }
  last_trail_update_time_ = -1e9;
}

Eigen::Vector3d TrajectoryVisualizer::GetBodyPointInWorld(
    const Context<double>& plant_context,
    BodyIndex body_index,
    const Eigen::Vector3d& local_point) const {
  const auto& body = plant_.get_body(body_index);
  RigidTransformd X_WB = plant_.EvalBodyPoseInWorld(plant_context, body);
  return X_WB * local_point;
}

EventStatus TrajectoryVisualizer::UpdateAndDrawTrails(
    const Context<double>& context) const {
  // Get plant state
  const auto& state_vector = get_plant_state_input().Eval(context);

  // Create a plant context with the current state
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Update load trail
  Eigen::Vector3d load_pos = GetBodyPointInWorld(
      *plant_context, load_body_index_, Eigen::Vector3d::Zero());
  load_trail_.push_back(load_pos);
  while (static_cast<int>(load_trail_.size()) > params_.load_max_trail_points) {
    load_trail_.pop_front();
  }

  // Update drone trails
  for (size_t i = 0; i < drone_body_indices_.size(); ++i) {
    Eigen::Vector3d drone_pos = GetBodyPointInWorld(
        *plant_context, drone_body_indices_[i], Eigen::Vector3d::Zero());
    drone_trails_[i].push_back(drone_pos);
    while (static_cast<int>(drone_trails_[i].size()) > params_.drone_max_trail_points) {
      drone_trails_[i].pop_front();
    }
  }

  // Draw load trail
  if (load_trail_.size() >= 2) {
    Eigen::Matrix3Xd points(3, load_trail_.size());
    int col = 0;
    for (const auto& pt : load_trail_) {
      points.col(col++) = pt;
    }
    meshcat_->SetLine(load_trail_path_, points,
                      params_.load_trail_width, params_.load_trail_color);
  }

  // Draw drone trails
  for (size_t i = 0; i < drone_trails_.size(); ++i) {
    if (drone_trails_[i].size() >= 2) {
      Eigen::Matrix3Xd points(3, drone_trails_[i].size());
      int col = 0;
      for (const auto& pt : drone_trails_[i]) {
        points.col(col++) = pt;
      }
      // Cycle through colors
      const auto& color = params_.drone_trail_colors[
          i % params_.drone_trail_colors.size()];
      meshcat_->SetLine(
          drone_trail_base_path_ + std::to_string(i),
          points, params_.drone_trail_width, color);
    }
  }

  return EventStatus::Succeeded();
}

void TrajectoryVisualizer::DrawReferenceTrajectory(
    const std::vector<Eigen::Vector3d>& waypoints) {
  if (!params_.show_reference_trajectory || waypoints.size() < 2) {
    return;
  }

  // Convert waypoints to matrix
  Eigen::Matrix3Xd points(3, waypoints.size());
  for (size_t i = 0; i < waypoints.size(); ++i) {
    points.col(i) = waypoints[i];
  }

  meshcat_->SetLine(reference_path_, points,
                    params_.reference_line_width, params_.reference_color);
}

void TrajectoryVisualizer::DrawReferenceTrajectoryFromFunction(
    std::function<Eigen::Vector3d(double)> trajectory_func,
    double start_time,
    double end_time,
    int num_samples) {
  if (!params_.show_reference_trajectory) {
    return;
  }

  std::vector<Eigen::Vector3d> waypoints;
  waypoints.reserve(num_samples);

  const double dt = (end_time - start_time) / (num_samples - 1);
  for (int i = 0; i < num_samples; ++i) {
    double t = start_time + i * dt;
    waypoints.push_back(trajectory_func(t));
  }

  DrawReferenceTrajectory(waypoints);
}

void TrajectoryVisualizer::ClearTrails() {
  load_trail_.clear();
  for (auto& trail : drone_trails_) {
    trail.clear();
  }

  // Delete from Meshcat
  meshcat_->Delete(load_trail_path_);
  for (size_t i = 0; i < drone_trails_.size(); ++i) {
    meshcat_->Delete(drone_trail_base_path_ + std::to_string(i));
  }
}

void TrajectoryVisualizer::ClearAll() {
  ClearTrails();
  meshcat_->Delete(reference_path_);
}

}  // namespace tether_lift
