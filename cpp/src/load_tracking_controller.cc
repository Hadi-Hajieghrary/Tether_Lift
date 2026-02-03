#include "load_tracking_controller.h"
#include <algorithm>
#include <cmath>

namespace tether_lift {

LoadTrackingController::LoadTrackingController(const Params& params)
    : params_(params) {

  // === Input ports ===

  // Drone state: [position(3), quaternion(4), velocity(3), angular_velocity(3)] = 13D
  drone_state_port_ = DeclareVectorInputPort("drone_state", 13).get_index();

  // Drone trajectory: [p_i^d(3), v_i^d(3), a_i^d(3)] = 9D
  drone_trajectory_port_ = DeclareVectorInputPort("drone_trajectory", 9).get_index();

  // Desired tension (scalar)
  desired_tension_port_ = DeclareVectorInputPort("desired_tension", 1).get_index();

  // Desired cable direction (3D unit vector)
  desired_cable_direction_port_ = DeclareVectorInputPort("desired_cable_direction", 3).get_index();

  // Actual cable direction (3D unit vector, from measurement/estimation)
  actual_cable_direction_port_ = DeclareVectorInputPort("actual_cable_direction", 3).get_index();

  // Actual tension (scalar, from measurement/estimation)
  actual_tension_port_ = DeclareVectorInputPort("actual_tension", 1).get_index();

  // === Continuous state for integral term ===
  DeclareContinuousState(3);  // Integral of position error (3D)

  // === Output ports ===

  // Thrust force vector in world frame (3D)
  thrust_force_port_ = DeclareVectorOutputPort(
      "thrust_force", 3,
      &LoadTrackingController::CalcThrustForce).get_index();

  // Thrust magnitude (scalar)
  thrust_magnitude_port_ = DeclareVectorOutputPort(
      "thrust_magnitude", 1,
      &LoadTrackingController::CalcThrustMagnitude).get_index();

  // Desired attitude as quaternion [w, x, y, z] (4D)
  desired_attitude_port_ = DeclareVectorOutputPort(
      "desired_attitude", 4,
      &LoadTrackingController::CalcDesiredAttitude).get_index();

  // Desired angular velocity (3D) - for attitude rate feedforward
  desired_angular_velocity_port_ = DeclareVectorOutputPort(
      "desired_angular_velocity", 3,
      &LoadTrackingController::CalcDesiredAngularVelocity).get_index();
}

void LoadTrackingController::SetDefaultState(
    const drake::systems::Context<double>& context,
    drake::systems::State<double>* state) const {
  // Initialize integral term to zero
  state->get_mutable_continuous_state()
      .get_mutable_vector()
      .SetFromVector(Eigen::Vector3d::Zero());
}

void LoadTrackingController::DoCalcTimeDerivatives(
    const drake::systems::Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {

  // Parse drone state
  const Eigen::VectorXd& state = get_drone_state_input().Eval(context);
  Eigen::Vector3d p = state.segment<3>(0);

  // Parse trajectory
  const Eigen::VectorXd& traj = get_drone_trajectory_input().Eval(context);
  Eigen::Vector3d p_des = traj.segment<3>(0);

  // Position error
  Eigen::Vector3d e = p - p_des;

  // Derivative of integral state is the position error
  // (with anti-windup: stop integrating if saturated)
  const Eigen::Vector3d& integral =
      context.get_continuous_state().get_vector().CopyToVector();

  Eigen::Vector3d integral_dot = e;

  // Anti-windup: limit integral growth
  for (int i = 0; i < 3; ++i) {
    if (std::abs(integral[i]) > params_.max_integral &&
        e[i] * integral[i] > 0) {
      integral_dot[i] = 0.0;  // Stop integrating in same direction
    }
  }

  derivatives->get_mutable_vector().SetFromVector(integral_dot);
}

Eigen::Vector3d LoadTrackingController::ComputeThrustVector(
    const drake::systems::Context<double>& context) const {

  // === Parse inputs ===

  // Drone state: [p(3), q(4), v(3), omega(3)]
  const Eigen::VectorXd& state = get_drone_state_input().Eval(context);
  Eigen::Vector3d p = state.segment<3>(0);
  // Quaternion at indices 3-6 (not used in this computation)
  Eigen::Vector3d v = state.segment<3>(7);

  // Drone trajectory: [p_des(3), v_des(3), a_des(3)]
  const Eigen::VectorXd& traj = get_drone_trajectory_input().Eval(context);
  Eigen::Vector3d p_des = traj.segment<3>(0);
  Eigen::Vector3d v_des = traj.segment<3>(3);
  Eigen::Vector3d a_des = traj.segment<3>(6);

  // Desired and actual cable values
  double T_des = get_desired_tension_input().Eval(context)[0];
  Eigen::Vector3d q_des = get_desired_cable_direction_input().Eval(context);
  double T_actual = get_actual_tension_input().Eval(context)[0];
  Eigen::Vector3d q_actual = get_actual_cable_direction_input().Eval(context);

  // Normalize cable directions
  if (q_des.norm() > 1e-6) q_des.normalize();
  else q_des = Eigen::Vector3d::UnitZ();

  if (q_actual.norm() > 1e-6) q_actual.normalize();
  else q_actual = Eigen::Vector3d::UnitZ();

  // === Compute tracking errors ===

  Eigen::Vector3d e = p - p_des;       // Position error
  Eigen::Vector3d e_dot = v - v_des;   // Velocity error

  // Integral of position error (from continuous state)
  Eigen::Vector3d e_int =
      context.get_continuous_state().get_vector().CopyToVector();

  // === Build feedback gain matrices ===

  Eigen::Matrix3d Kp = params_.Kp.asDiagonal();
  Eigen::Matrix3d Kd = params_.Kd.asDiagonal();
  Eigen::Matrix3d Ki = params_.Ki.asDiagonal();

  // === Gravity vector ===
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // === Compute control force (Equation 21) ===
  //
  // f_i^d = m_i·(a_i^d + g·e₃) + T_i·q_i - Kp·e - Kd·ė - Ki·∫e
  //
  // The cable tension term compensates for the force the cable exerts on the drone.
  // We can use either desired (T_des, q_des) or actual (T_actual, q_actual) values.

  // Feedforward: mass times (desired acceleration + gravity)
  Eigen::Vector3d f_feedforward = params_.mass * (a_des + params_.gravity * e3);

  // Cable tension compensation
  // The cable pulls the drone toward the load. To maintain position,
  // the drone must thrust to counteract this force.
  // Cable force on drone is -T·q (pointing from drone toward load = -q direction)
  // So the drone must generate +T·q in its thrust.
  Eigen::Vector3d f_cable;
  if (params_.use_actual_tension_feedforward) {
    // Use actual measured/estimated values for better disturbance rejection
    f_cable = params_.cable_compensation_gain * T_actual * q_actual;
  } else {
    // Use desired values (model-based)
    f_cable = params_.cable_compensation_gain * T_des * q_des;
  }

  // Feedback: PID on position error
  Eigen::Vector3d f_feedback = -Kp * e - Kd * e_dot - Ki * e_int;

  // Total desired thrust
  Eigen::Vector3d f_total = f_feedforward + f_cable + f_feedback;

  // === Apply saturation ===

  double thrust_norm = f_total.norm();
  if (thrust_norm > params_.max_thrust) {
    f_total = f_total * (params_.max_thrust / thrust_norm);
  }
  if (thrust_norm < params_.min_thrust) {
    // Ensure positive thrust (drone can't push down)
    f_total = params_.min_thrust * e3;
  }

  return f_total;
}

Eigen::Quaterniond LoadTrackingController::ThrustVectorToQuaternion(
    const Eigen::Vector3d& thrust,
    double yaw_des) const {

  // The desired thrust direction defines the body z-axis
  Eigen::Vector3d z_body = thrust.normalized();

  // Desired yaw defines the projection of body x-axis
  Eigen::Vector3d x_c(std::cos(yaw_des), std::sin(yaw_des), 0.0);

  // Body y-axis: perpendicular to z_body and in the plane with x_c
  Eigen::Vector3d y_body = z_body.cross(x_c);
  if (y_body.norm() < 1e-6) {
    // z_body is parallel to x_c, use alternative
    y_body = z_body.cross(Eigen::Vector3d::UnitY());
  }
  y_body.normalize();

  // Body x-axis: complete the right-handed frame
  Eigen::Vector3d x_body = y_body.cross(z_body);
  x_body.normalize();

  // Build rotation matrix (world to body)
  Eigen::Matrix3d R;
  R.col(0) = x_body;
  R.col(1) = y_body;
  R.col(2) = z_body;

  return Eigen::Quaterniond(R);
}

void LoadTrackingController::CalcThrustForce(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  output->get_mutable_value() = ComputeThrustVector(context);
}

void LoadTrackingController::CalcThrustMagnitude(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  double mag = ComputeThrustVector(context).norm();
  output->get_mutable_value() << mag;
}

void LoadTrackingController::CalcDesiredAttitude(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  Eigen::Vector3d thrust = ComputeThrustVector(context);

  // TODO: Extract desired yaw from trajectory or input
  double yaw_des = 0.0;

  Eigen::Quaterniond q = ThrustVectorToQuaternion(thrust, yaw_des);

  // Output as [w, x, y, z]
  output->get_mutable_value() << q.w(), q.x(), q.y(), q.z();
}

void LoadTrackingController::CalcDesiredAngularVelocity(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {

  // For now, return zero angular velocity (attitude tracking only)
  // TODO: Compute from differentiation of desired attitude for feedforward
  output->get_mutable_value() = Eigen::Vector3d::Zero();
}

}  // namespace tether_lift
