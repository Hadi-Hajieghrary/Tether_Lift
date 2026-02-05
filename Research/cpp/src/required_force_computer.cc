#include "required_force_computer.h"
#include <algorithm>
#include <cmath>

namespace tether_lift {

RequiredForceComputer::RequiredForceComputer(const Params& params)
    : params_(params) {
  // Input: load trajectory [p_L^d (3), v_L^d (3), a_L^d (3)] = 9D
  load_trajectory_port_ = DeclareVectorInputPort("load_trajectory", 9).get_index();

  // Input: load state [p_L (3), v_L (3)] = 6D
  load_state_port_ = DeclareVectorInputPort("load_state", 6).get_index();

  // Input: theta_hat (scalar)
  theta_hat_port_ = DeclareVectorInputPort("theta_hat", 1).get_index();

  // Input: number of drones N (scalar, can be integer cast)
  num_drones_port_ = DeclareVectorInputPort("num_drones", 1).get_index();

  // Output: required force F_req (3D)
  required_force_port_ = DeclareVectorOutputPort(
      "required_force", 3,
      &RequiredForceComputer::CalcRequiredForce).get_index();

  // Output: estimated load mass m̂_L (scalar)
  estimated_mass_port_ = DeclareVectorOutputPort(
      "estimated_mass", 1,
      &RequiredForceComputer::CalcEstimatedMass).get_index();
}

double RequiredForceComputer::ComputeEstimatedMass(
    const drake::systems::Context<double>& context) const {
  // Get inputs
  const auto& theta_hat_vec = get_theta_hat_input().Eval(context);
  const auto& num_drones_vec = get_num_drones_input().Eval(context);

  double theta_hat = theta_hat_vec[0];
  double N = num_drones_vec[0];

  // m̂_L = N · θ̂ where θ̂ = m_L / N
  double mass_estimate = N * theta_hat;

  // Apply minimum mass floor for safety
  return std::max(mass_estimate, params_.min_mass_estimate);
}

void RequiredForceComputer::CalcRequiredForce(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  // Parse load trajectory input
  const Eigen::VectorXd& trajectory = get_load_trajectory_input().Eval(context);
  Eigen::Vector3d p_L_des = trajectory.segment<3>(0);  // Desired position
  Eigen::Vector3d v_L_des = trajectory.segment<3>(3);  // Desired velocity
  Eigen::Vector3d a_L_des = trajectory.segment<3>(6);  // Desired acceleration (feedforward)

  // Parse load state input
  const Eigen::VectorXd& state = get_load_state_input().Eval(context);
  Eigen::Vector3d p_L = state.segment<3>(0);  // Actual position
  Eigen::Vector3d v_L = state.segment<3>(3);  // Actual velocity

  // Compute tracking errors
  Eigen::Vector3d e_L = p_L - p_L_des;    // Position error
  Eigen::Vector3d e_dot_L = v_L - v_L_des; // Velocity error

  // Get estimated load mass
  double m_hat_L = ComputeEstimatedMass(context);

  // Gravity unit vector (pointing up in inertial frame)
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // Construct diagonal gain matrices
  Eigen::Matrix3d Kp = params_.Kp.asDiagonal();
  Eigen::Matrix3d Kd = params_.Kd.asDiagonal();

  // Equation (5): F_req = m̂_L * (ä_L^d + g·e₃ - Kp·e_L - Kd·ė_L)
  // This is the total force that ALL cables combined must apply to the load
  Eigen::Vector3d F_req = m_hat_L * (
      a_L_des +                        // Feedforward acceleration
      params_.gravity * e3 -           // Gravity compensation
      Kp * e_L -                        // Position feedback
      Kd * e_dot_L                      // Velocity feedback
  );

  // Apply saturation for safety
  double force_magnitude = F_req.norm();
  if (force_magnitude > params_.max_force_magnitude) {
    F_req = F_req * (params_.max_force_magnitude / force_magnitude);
  }

  // Set output
  output->get_mutable_value() = F_req;
}

void RequiredForceComputer::CalcEstimatedMass(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  double mass = ComputeEstimatedMass(context);
  output->get_mutable_value() << mass;
}

}  // namespace tether_lift
