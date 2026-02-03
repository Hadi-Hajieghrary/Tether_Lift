#include "decentralized_drone_controller.h"
#include <algorithm>
#include <cmath>

namespace tether_lift {

DecentralizedDroneController::DecentralizedDroneController(const Params& params)
    : params_(params) {

  // === Input ports ===

  // Drone state: [position(3), quaternion(4), velocity(3), angular_velocity(3)] = 13D
  drone_state_port_ = DeclareVectorInputPort("drone_state", 13).get_index();

  // Load state: [p_L(3), v_L(3)] = 6D (from shared GPS broadcast)
  load_state_port_ = DeclareVectorInputPort("load_state", 6).get_index();

  // Load trajectory: [p_L^d(3), v_L^d(3), a_L^d(3)] = 9D (shared broadcast)
  load_trajectory_port_ = DeclareVectorInputPort("load_trajectory", 9).get_index();

  // Adaptive parameter θ̂_i (scalar) - from this drone's AdaptiveLoadEstimator
  theta_hat_port_ = DeclareVectorInputPort("theta_hat", 1).get_index();

  // Measured cable tension T_i (scalar)
  cable_tension_port_ = DeclareVectorInputPort("cable_tension", 1).get_index();

  // Measured cable direction q_i (3D unit vector, pointing from load to drone)
  cable_direction_port_ = DeclareVectorInputPort("cable_direction", 3).get_index();

  // === Output ports ===

  // Thrust force in world frame (3D)
  thrust_force_port_ = DeclareVectorOutputPort(
      "thrust_force", 3,
      &DecentralizedDroneController::CalcThrustForce).get_index();

  // Desired drone position (3D) - for logging
  desired_position_port_ = DeclareVectorOutputPort(
      "desired_position", 3,
      &DecentralizedDroneController::CalcDesiredPosition).get_index();

  // Desired cable direction (3D) - for logging
  desired_direction_port_ = DeclareVectorOutputPort(
      "desired_direction", 3,
      &DecentralizedDroneController::CalcDesiredDirection).get_index();

  // Desired tension (scalar) - for logging
  desired_tension_port_ = DeclareVectorOutputPort(
      "desired_tension", 1,
      &DecentralizedDroneController::CalcDesiredTension).get_index();
}

Eigen::Vector3d DecentralizedDroneController::ComputeDesiredCableForce(
    const drake::systems::Context<double>& context) const {

  // === Get load state (shared broadcast) ===
  const Eigen::VectorXd& load_state = get_load_state_input().Eval(context);
  Eigen::Vector3d p_L = load_state.segment<3>(0);  // Load position
  Eigen::Vector3d v_L = load_state.segment<3>(3);  // Load velocity

  // === Get load trajectory (shared broadcast) ===
  const Eigen::VectorXd& load_traj = get_load_trajectory_input().Eval(context);
  Eigen::Vector3d p_L_des = load_traj.segment<3>(0);  // Desired position
  Eigen::Vector3d v_L_des = load_traj.segment<3>(3);  // Desired velocity
  Eigen::Vector3d a_L_des = load_traj.segment<3>(6);  // Desired acceleration (feedforward)

  // === Get adaptive parameter (local to this drone) ===
  double theta_hat = get_theta_hat_input().Eval(context)[0];

  // Safety: ensure positive theta_hat
  theta_hat = std::max(theta_hat, 0.1);

  // === Compute load tracking errors ===
  Eigen::Vector3d e_L = p_L - p_L_des;      // Position error
  Eigen::Vector3d e_dot_L = v_L - v_L_des;  // Velocity error

  // === Build gain matrices ===
  Eigen::Matrix3d Kp = params_.Kp_load.asDiagonal();
  Eigen::Matrix3d Kd = params_.Kd_load.asDiagonal();

  // === Gravity vector ===
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // === Core control law (Equation 5, but using local θ̂_i only) ===
  //
  // F_i^d = θ̂_i × (ä_L^d + g·e₃ - Kp·e_L - Kd·ė_L)
  //
  // This is the force THIS drone should apply via its cable.
  // With N drones each running this, total force = m_L × (control term)
  //
  Eigen::Vector3d F_i_des = theta_hat * (
      a_L_des +                    // Feedforward acceleration
      params_.gravity * e3 -       // Gravity compensation
      Kp * e_L -                   // Position feedback
      Kd * e_dot_L                 // Velocity feedback
  );

  return F_i_des;
}

Eigen::Vector3d DecentralizedDroneController::ComputeDesiredDronePosition(
    const Eigen::Vector3d& load_position,
    const Eigen::Vector3d& desired_cable_direction) const {

  // p_i^d = p_L + a_i^L + ℓ_i · q_i^d
  //
  // - p_L: current load position
  // - a_i^L: attachment point offset (this drone's connection point on load)
  // - ℓ_i: cable length
  // - q_i^d: desired cable direction (unit vector from load to drone)

  return load_position +
         params_.attachment_point +
         params_.cable_length * desired_cable_direction;
}

void DecentralizedDroneController::CalcThrustForce(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  // === Get drone state ===
  const Eigen::VectorXd& drone_state = get_drone_state_input().Eval(context);
  Eigen::Vector3d p_i = drone_state.segment<3>(0);   // Drone position
  Eigen::Vector3d v_i = drone_state.segment<3>(7);   // Drone velocity

  // === Get load state for position computation ===
  const Eigen::VectorXd& load_state = get_load_state_input().Eval(context);
  Eigen::Vector3d p_L = load_state.segment<3>(0);
  Eigen::Vector3d v_L = load_state.segment<3>(3);

  // === Get load trajectory for velocity reference ===
  const Eigen::VectorXd& load_traj = get_load_trajectory_input().Eval(context);
  Eigen::Vector3d v_L_des = load_traj.segment<3>(3);
  Eigen::Vector3d a_L_des = load_traj.segment<3>(6);

  // === Get cable measurements ===
  double T_i = get_cable_tension_input().Eval(context)[0];
  Eigen::Vector3d q_i = get_cable_direction_input().Eval(context);
  if (q_i.norm() > 1e-6) q_i.normalize();
  else q_i = Eigen::Vector3d::UnitZ();

  // === Compute desired cable force ===
  Eigen::Vector3d F_i_des = ComputeDesiredCableForce(context);

  // === Extract desired tension and direction ===
  double T_i_des = F_i_des.norm();
  T_i_des = std::max(T_i_des, params_.min_tension);

  Eigen::Vector3d q_i_des;
  if (F_i_des.norm() > 1e-6) {
    q_i_des = F_i_des.normalized();
  } else {
    // Default: point upward with some spread
    q_i_des = Eigen::Vector3d::UnitZ();
  }

  // === Compute desired drone position ===
  Eigen::Vector3d p_i_des = ComputeDesiredDronePosition(p_L, q_i_des);

  // === Compute desired drone velocity (quasi-static: follows load) ===
  Eigen::Vector3d v_i_des = v_L_des;

  // === Drone tracking errors ===
  Eigen::Vector3d e_i = p_i - p_i_des;
  Eigen::Vector3d e_dot_i = v_i - v_i_des;

  // === Build drone tracking gains ===
  Eigen::Matrix3d Kp_drone = params_.Kp_drone.asDiagonal();
  Eigen::Matrix3d Kd_drone = params_.Kd_drone.asDiagonal();

  // === Gravity vector ===
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // === Full thrust computation ===
  //
  // f_i = m_i·(a_i^d + g·e₃) + T_i·q_i - Kp·e_i - Kd·ė_i
  //
  // Components:
  // 1. Feedforward: mass × (desired acceleration + gravity)
  // 2. Cable compensation: counteract the cable pulling the drone
  // 3. Position feedback: correct drone position error
  // 4. Velocity feedback: correct drone velocity error

  // Feedforward (using load acceleration as reference for drone)
  Eigen::Vector3d f_feedforward = params_.drone_mass * (a_L_des + params_.gravity * e3);

  // Cable compensation: cable pulls drone toward load with force T_i·q_i
  // Drone must thrust to counteract this
  Eigen::Vector3d f_cable = params_.cable_compensation_gain * T_i * q_i;

  // Feedback
  Eigen::Vector3d f_feedback = -Kp_drone * e_i - Kd_drone * e_dot_i;

  // Total thrust
  Eigen::Vector3d f_total = f_feedforward + f_cable + f_feedback;

  // === Apply saturation ===
  double thrust_norm = f_total.norm();
  if (thrust_norm > params_.max_thrust) {
    f_total = f_total * (params_.max_thrust / thrust_norm);
  }
  if (thrust_norm < params_.min_thrust) {
    f_total = params_.min_thrust * e3;
  }

  output->get_mutable_value() = f_total;
}

void DecentralizedDroneController::CalcDesiredPosition(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  // Get load position
  const Eigen::VectorXd& load_state = get_load_state_input().Eval(context);
  Eigen::Vector3d p_L = load_state.segment<3>(0);

  // Compute desired cable direction
  Eigen::Vector3d F_i_des = ComputeDesiredCableForce(context);
  Eigen::Vector3d q_i_des;
  if (F_i_des.norm() > 1e-6) {
    q_i_des = F_i_des.normalized();
  } else {
    q_i_des = Eigen::Vector3d::UnitZ();
  }

  // Compute desired drone position
  output->get_mutable_value() = ComputeDesiredDronePosition(p_L, q_i_des);
}

void DecentralizedDroneController::CalcDesiredDirection(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d F_i_des = ComputeDesiredCableForce(context);

  Eigen::Vector3d q_i_des;
  if (F_i_des.norm() > 1e-6) {
    q_i_des = F_i_des.normalized();
  } else {
    q_i_des = Eigen::Vector3d::UnitZ();
  }

  output->get_mutable_value() = q_i_des;
}

void DecentralizedDroneController::CalcDesiredTension(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d F_i_des = ComputeDesiredCableForce(context);
  double T_i_des = std::max(F_i_des.norm(), params_.min_tension);

  output->get_mutable_value() << T_i_des;
}

}  // namespace tether_lift
