#include "drone_trajectory_mapper.h"
#include <cmath>

namespace tether_lift {

void DroneTrajectoryMapper::Params::InitializeDefaultAttachments() {
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
    cable_lengths.push_back(default_cable_length);
  }
}

DroneTrajectoryMapper::DroneTrajectoryMapper(const Params& params)
    : params_(params) {
  // Initialize default attachments if not provided
  if (params_.attachments.empty()) {
    params_.InitializeDefaultAttachments();
  }

  // Ensure cable lengths vector is sized correctly
  if (params_.cable_lengths.empty()) {
    params_.cable_lengths.resize(params_.num_drones, params_.default_cable_length);
  }

  const int N = params_.num_drones;

  // Input: load trajectory [p_L^d (3), v_L^d (3), a_L^d (3), j_L^d (3)] = 12D
  // (jerk is optional, used for computing q̇_i^d in full dynamics mode)
  load_trajectory_port_ = DeclareVectorInputPort("load_trajectory", 12).get_index();

  // Input: desired cable directions (N×3D = 3N)
  desired_directions_port_ = DeclareVectorInputPort(
      "desired_directions", 3 * N).get_index();

  // Input: load orientation as quaternion [w, x, y, z] (4D)
  load_orientation_port_ = DeclareVectorInputPort("load_orientation", 4).get_index();

  // Output: drone trajectories (N×9D = 9N: [p_i^d, v_i^d, a_i^d] per drone)
  drone_trajectories_port_ = DeclareVectorOutputPort(
      "drone_trajectories", 9 * N,
      &DroneTrajectoryMapper::CalcDroneTrajectories).get_index();
}

Eigen::Vector3d DroneTrajectoryMapper::RotateToWorld(
    const Eigen::Vector3d& v_load,
    const Eigen::Quaterniond& q_load) const {
  return q_load * v_load;
}

void DroneTrajectoryMapper::CalcDroneTrajectories(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  const int N = params_.num_drones;

  // Parse load trajectory
  const Eigen::VectorXd& traj = get_load_trajectory_input().Eval(context);
  Eigen::Vector3d p_L_des = traj.segment<3>(0);   // Desired load position
  Eigen::Vector3d v_L_des = traj.segment<3>(3);   // Desired load velocity
  Eigen::Vector3d a_L_des = traj.segment<3>(6);   // Desired load acceleration
  // Eigen::Vector3d j_L_des = traj.segment<3>(9);   // Desired load jerk (for full dynamics)

  // Parse desired cable directions
  const Eigen::VectorXd& directions = get_desired_directions_input().Eval(context);

  // Parse load orientation quaternion [w, x, y, z]
  const Eigen::VectorXd& quat_vec = get_load_orientation_input().Eval(context);
  Eigen::Quaterniond q_load(quat_vec[0], quat_vec[1], quat_vec[2], quat_vec[3]);
  q_load.normalize();

  // Compute trajectory for each drone
  Eigen::VectorXd out(9 * N);

  for (int i = 0; i < N; ++i) {
    // Get attachment point in load frame
    const Eigen::Vector3d& a_i_L = params_.attachments[i].position;

    // Rotate attachment to world frame
    Eigen::Vector3d a_i_W = RotateToWorld(a_i_L, q_load);

    // Get cable length and direction
    double ell_i = params_.cable_lengths[i];
    Eigen::Vector3d q_i = directions.segment<3>(3 * i).normalized();

    // Equation (17): p_i^d = p_L^d + a_i^L + ℓ_i · q_i^d
    // Cable direction q_i points from load toward drone
    Eigen::Vector3d p_i_des = p_L_des + a_i_W + ell_i * q_i;

    // For quasi-static assumption (q̇ ≈ 0, q̈ ≈ 0):
    // Equation (18): v_i^d = v_L^d + ℓ_i · q̇_i^d ≈ v_L^d
    // Equation (19): a_i^d = a_L^d + ℓ_i · q̈_i^d ≈ a_L^d
    Eigen::Vector3d v_i_des = v_L_des;
    Eigen::Vector3d a_i_des = a_L_des;

    // TODO: For full dynamics mode, compute q̇_i^d and q̈_i^d from:
    // q̇ = (I - q·qᵀ) · ω_cable / ||ρ||
    // where ω_cable depends on how the required force direction changes

    // Pack into output: [p, v, a]
    out.segment<3>(9 * i + 0) = p_i_des;
    out.segment<3>(9 * i + 3) = v_i_des;
    out.segment<3>(9 * i + 6) = a_i_des;
  }

  output->get_mutable_value() = out;
}

// ============================================================================
// DroneTrajectoryDemux
// ============================================================================

DroneTrajectoryDemux::DroneTrajectoryDemux(int num_drones)
    : num_drones_(num_drones) {
  // Input: stacked trajectories (N × 9D)
  DeclareVectorInputPort("drone_trajectories", 9 * num_drones);

  // Create N output ports, one per drone
  for (int i = 0; i < num_drones; ++i) {
    DeclareVectorOutputPort(
        "drone_" + std::to_string(i) + "_trajectory", 9,
        [this, i](const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) {
          this->CalcDroneTrajectory(context, output, i);
        });
  }
}

void DroneTrajectoryDemux::CalcDroneTrajectory(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output,
    int drone_index) const {
  const Eigen::VectorXd& all_trajectories = get_input().Eval(context);
  output->get_mutable_value() = all_trajectories.segment<9>(9 * drone_index);
}

}  // namespace tether_lift
