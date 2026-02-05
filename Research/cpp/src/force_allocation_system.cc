#include "force_allocation_system.h"
#include <algorithm>
#include <cmath>

namespace tether_lift {

void ForceAllocationSystem::Params::InitializeDefaultAttachments() {
  attachments.clear();
  cable_lengths.clear();

  // Create evenly-spaced attachment points in a circle
  const double angle_step = 2.0 * M_PI / num_drones;

  for (int i = 0; i < num_drones; ++i) {
    double angle = i * angle_step;
    AttachmentPoint ap;
    ap.position = Eigen::Vector3d(
        default_attachment_radius * std::cos(angle),
        default_attachment_radius * std::sin(angle),
        0.0  // Attachments at load center height
    );
    attachments.push_back(ap);
    cable_lengths.push_back(3.0);  // Default 3m cables
  }
}

ForceAllocationSystem::ForceAllocationSystem(const Params& params)
    : params_(params) {
  // Initialize default attachments if not provided
  if (params_.attachments.empty()) {
    params_.InitializeDefaultAttachments();
  }

  // Ensure cable lengths vector is sized correctly
  if (params_.cable_lengths.empty()) {
    params_.cable_lengths.resize(params_.num_drones, 3.0);
  }

  const int N = params_.num_drones;

  // Input: required total cable force (3D)
  required_force_port_ = DeclareVectorInputPort("required_force", 3).get_index();

  // Input: load position (3D)
  load_position_port_ = DeclareVectorInputPort("load_position", 3).get_index();

  // Output: allocated forces for each drone (N×3D = 3N elements)
  allocated_forces_port_ = DeclareVectorOutputPort(
      "allocated_forces", 3 * N,
      &ForceAllocationSystem::CalcAllocatedForces).get_index();

  // Output: desired tensions (N elements)
  desired_tensions_port_ = DeclareVectorOutputPort(
      "desired_tensions", N,
      &ForceAllocationSystem::CalcDesiredTensions).get_index();

  // Output: desired cable directions (N×3D = 3N elements)
  desired_directions_port_ = DeclareVectorOutputPort(
      "desired_directions", 3 * N,
      &ForceAllocationSystem::CalcDesiredDirections).get_index();
}

Eigen::Vector3d ForceAllocationSystem::ComputeSpreadingDirection(int drone_index) const {
  // Get attachment point in load frame
  const Eigen::Vector3d& a_i = params_.attachments[drone_index].position;

  // Compute horizontal projection
  Eigen::Vector3d horizontal(a_i.x(), a_i.y(), 0.0);
  double norm = horizontal.norm();

  if (norm < 1e-6) {
    // Attachment at center, use radial direction based on index
    double angle = 2.0 * M_PI * drone_index / params_.num_drones;
    return Eigen::Vector3d(std::cos(angle), std::sin(angle), 0.0);
  }

  return horizontal / norm;
}

ForceAllocationSystem::AllocationResult ForceAllocationSystem::ComputeAllocation(
    const drake::systems::Context<double>& context) const {
  const int N = params_.num_drones;
  AllocationResult result;
  result.forces.resize(N);
  result.tensions.resize(N);
  result.directions.resize(N);

  // Get required total force
  const Eigen::Vector3d F_req = get_required_force_input().Eval(context).head<3>();

  // Equal share per drone
  Eigen::Vector3d F_share = F_req / static_cast<double>(N);

  // Compute allocation for each drone
  for (int i = 0; i < N; ++i) {
    // Get horizontal spreading direction
    Eigen::Vector3d e_r_i = ComputeSpreadingDirection(i);

    // Equation (11): μ_i^d = F_req/N + α·ê_r,i
    // The spreading term ensures cables stay taut and spread out
    Eigen::Vector3d mu_i = F_share + params_.alpha * e_r_i;

    // Compute tension magnitude
    double T_i = mu_i.norm();

    // Apply tension bounds
    T_i = std::clamp(T_i, params_.T_min, params_.T_max);

    // Compute cable direction (pointing from load to drone)
    Eigen::Vector3d q_i;
    if (T_i > 1e-6) {
      q_i = mu_i / mu_i.norm();  // Use original mu_i norm for direction
    } else {
      // Fallback: point mostly upward with some spread
      q_i = Eigen::Vector3d(e_r_i.x() * 0.3, e_r_i.y() * 0.3, 0.9).normalized();
    }

    // Store results
    result.forces[i] = mu_i;
    result.tensions[i] = T_i;
    result.directions[i] = q_i;
  }

  return result;
}

void ForceAllocationSystem::CalcAllocatedForces(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  AllocationResult result = ComputeAllocation(context);

  Eigen::VectorXd out(3 * params_.num_drones);
  for (int i = 0; i < params_.num_drones; ++i) {
    out.segment<3>(3 * i) = result.forces[i];
  }
  output->get_mutable_value() = out;
}

void ForceAllocationSystem::CalcDesiredTensions(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  AllocationResult result = ComputeAllocation(context);

  Eigen::VectorXd out(params_.num_drones);
  for (int i = 0; i < params_.num_drones; ++i) {
    out[i] = result.tensions[i];
  }
  output->get_mutable_value() = out;
}

void ForceAllocationSystem::CalcDesiredDirections(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  AllocationResult result = ComputeAllocation(context);

  Eigen::VectorXd out(3 * params_.num_drones);
  for (int i = 0; i < params_.num_drones; ++i) {
    out.segment<3>(3 * i) = result.directions[i];
  }
  output->get_mutable_value() = out;
}

}  // namespace tether_lift
