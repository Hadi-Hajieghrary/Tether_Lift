#pragma once

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Dense>
#include <vector>

namespace tether_lift {

/**
 * @brief Maps desired load trajectory to individual drone trajectories.
 *
 * Implements Equations (17-19) from the mathematical framework:
 *
 * Position (Eq. 17):
 *   p_i^d = p_L^d + a_i^L + ℓ_i · q_i^d
 *
 * Velocity (Eq. 18):
 *   v_i^d = v_L^d + ℓ_i · q̇_i^d
 *
 * Acceleration (Eq. 19):
 *   a_i^d = a_L^d + ℓ_i · q̈_i^d
 *
 * where:
 *   - p_L^d, v_L^d, a_L^d are desired load position/velocity/acceleration
 *   - a_i^L is the attachment point offset in load frame
 *   - ℓ_i is the cable length for drone i
 *   - q_i^d is the desired cable direction (from ForceAllocationSystem)
 *   - q̇_i^d, q̈_i^d are computed from cable dynamics
 *
 * For quasi-static motion, we approximate q̇_i^d ≈ 0 and q̈_i^d ≈ 0,
 * which is valid for slow maneuvers. For dynamic trajectories,
 * the full derivatives should be computed.
 *
 * Input ports:
 *   0: load_trajectory (12D: p_L^d, v_L^d, a_L^d, j_L^d - with jerk for q̇ computation)
 *   1: desired_directions (N×3D: q_i^d from ForceAllocationSystem)
 *   2: load_orientation (4D: quaternion - for rotating a_i^L to world frame)
 *
 * Output ports:
 *   0: drone_trajectories (N×9D: [p_i^d, v_i^d, a_i^d] for each drone)
 */
class DroneTrajectoryMapper final : public drake::systems::LeafSystem<double> {
 public:
  struct AttachmentPoint {
    Eigen::Vector3d position;  // a_i^L in load frame
  };

  struct Params {
    int num_drones = 4;

    // Attachment points in load frame
    std::vector<AttachmentPoint> attachments;

    // Cable lengths
    std::vector<double> cable_lengths;

    // Default attachment radius for automatic configuration
    double default_attachment_radius = 0.3;  // m

    // Default cable length
    double default_cable_length = 3.0;  // m

    // Enable full dynamics (compute q̇_i^d and q̈_i^d)
    // If false, assumes quasi-static: q̇ = q̈ = 0
    bool enable_full_dynamics = false;

    // Time constant for cable direction filtering (smoothing)
    double direction_filter_tau = 0.1;  // s

    // Initialize with default attachments for N drones
    void InitializeDefaultAttachments();
  };

  explicit DroneTrajectoryMapper(const Params& params);

  // Input port getters
  const drake::systems::InputPort<double>& get_load_trajectory_input() const {
    return get_input_port(load_trajectory_port_);
  }

  const drake::systems::InputPort<double>& get_desired_directions_input() const {
    return get_input_port(desired_directions_port_);
  }

  const drake::systems::InputPort<double>& get_load_orientation_input() const {
    return get_input_port(load_orientation_port_);
  }

  // Output port getters
  const drake::systems::OutputPort<double>& get_drone_trajectories_output() const {
    return get_output_port(drone_trajectories_port_);
  }

  // Get individual drone trajectory (convenience for wiring)
  // Returns port index for drone i's 9D trajectory
  int GetDroneTrajectoryStartIndex(int drone_index) const {
    return 9 * drone_index;
  }

  // Get number of drones
  int num_drones() const { return params_.num_drones; }

  // Parameter accessors
  const Params& params() const { return params_; }

 private:
  void CalcDroneTrajectories(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Rotate vector from load frame to world frame using quaternion
  Eigen::Vector3d RotateToWorld(
      const Eigen::Vector3d& v_load,
      const Eigen::Quaterniond& q_load) const;

  Params params_;

  // Port indices
  int load_trajectory_port_{};
  int desired_directions_port_{};
  int load_orientation_port_{};
  int drone_trajectories_port_{};
};

/**
 * @brief Demultiplexer for drone trajectories.
 *
 * Takes the stacked N×9D drone trajectories and outputs individual 9D
 * trajectories for each drone. This simplifies wiring to N separate controllers.
 */
class DroneTrajectoryDemux final : public drake::systems::LeafSystem<double> {
 public:
  explicit DroneTrajectoryDemux(int num_drones);

  const drake::systems::InputPort<double>& get_input() const {
    return get_input_port(0);
  }

  const drake::systems::OutputPort<double>& get_drone_output(int drone_index) const {
    return get_output_port(drone_index);
  }

  int num_drones() const { return num_drones_; }

 private:
  void CalcDroneTrajectory(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output,
      int drone_index) const;

  int num_drones_;
};

}  // namespace tether_lift
