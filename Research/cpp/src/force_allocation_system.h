#pragma once

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Dense>
#include <vector>

namespace tether_lift {

/**
 * @brief Allocates the required total cable force among N drones.
 *
 * Implements Equation (11) from the mathematical framework:
 *   μ_i^d = F_req/N + α·ê_r,i
 *
 * where:
 *   - F_req is the total required cable force (from RequiredForceComputer)
 *   - N is the number of drones
 *   - α is a spreading coefficient ensuring cables don't collapse
 *   - ê_r,i is the horizontal unit vector from load to attachment point i
 *
 * The system then computes:
 *   - T_i^d = ||μ_i^d|| (desired tension magnitude)
 *   - q_i^d = μ_i^d / T_i^d (desired cable direction from load to drone)
 *
 * And verifies feasibility (Equation 9):
 *   T_i^d ≥ T_min (minimum tension constraint)
 *
 * Input ports:
 *   0: required_force (3D: F_req from RequiredForceComputer)
 *   1: load_position (3D: p_L actual load position)
 *
 * Output ports:
 *   0: allocated_forces (N×3D: μ_i^d vectors stacked)
 *   1: desired_tensions (N×1D: T_i^d scalars)
 *   2: desired_directions (N×3D: q_i^d unit vectors stacked)
 */
class ForceAllocationSystem final : public drake::systems::LeafSystem<double> {
 public:
  struct AttachmentPoint {
    Eigen::Vector3d position;  // a_i^L in load frame
  };

  struct Params {
    int num_drones = 4;

    // Attachment points in load frame (default: square formation)
    std::vector<AttachmentPoint> attachments;

    // Spreading coefficient (N·m) - prevents cable collapse
    double alpha = 5.0;

    // Minimum tension constraint (N)
    double T_min = 1.0;

    // Maximum tension constraint (N)
    double T_max = 100.0;

    // Cable lengths (uniform by default)
    std::vector<double> cable_lengths;

    // Default attachment radius for automatic configuration
    double default_attachment_radius = 0.3;  // m

    // Initialize with default attachments for N drones
    void InitializeDefaultAttachments();
  };

  explicit ForceAllocationSystem(const Params& params);

  // Input port getters
  const drake::systems::InputPort<double>& get_required_force_input() const {
    return get_input_port(required_force_port_);
  }

  const drake::systems::InputPort<double>& get_load_position_input() const {
    return get_input_port(load_position_port_);
  }

  // Output port getters
  const drake::systems::OutputPort<double>& get_allocated_forces_output() const {
    return get_output_port(allocated_forces_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_tensions_output() const {
    return get_output_port(desired_tensions_port_);
  }

  const drake::systems::OutputPort<double>& get_desired_directions_output() const {
    return get_output_port(desired_directions_port_);
  }

  // Get number of drones
  int num_drones() const { return params_.num_drones; }

  // Parameter accessors
  const Params& params() const { return params_; }

 private:
  void CalcAllocatedForces(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredTensions(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcDesiredDirections(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  // Compute horizontal spreading direction for drone i
  Eigen::Vector3d ComputeSpreadingDirection(int drone_index) const;

  // Core allocation computation (shared by output methods)
  struct AllocationResult {
    std::vector<Eigen::Vector3d> forces;      // μ_i^d
    std::vector<double> tensions;              // T_i^d
    std::vector<Eigen::Vector3d> directions;  // q_i^d
  };

  AllocationResult ComputeAllocation(
      const drake::systems::Context<double>& context) const;

  Params params_;

  // Port indices
  int required_force_port_{};
  int load_position_port_{};
  int allocated_forces_port_{};
  int desired_tensions_port_{};
  int desired_directions_port_{};
};

}  // namespace tether_lift
