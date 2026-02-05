/// @file gpac_load_tracking_controller.cc
/// @brief GPAC-enhanced load tracking controller implementation.

#include "gpac_load_tracking_controller.h"

#include <algorithm>
#include <cmath>

namespace tether_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using quad_rope_lift::gpac::CableDirectionError;
using quad_rope_lift::gpac::ProjectToTangentS2;
using quad_rope_lift::gpac::PsiS2;

GPACLoadTrackingController::GPACLoadTrackingController(
    const GPACLoadTrackingParams& params)
    : params_(params) {

  // === Input ports ===

  drone_state_port_ = DeclareVectorInputPort("drone_state", 13).get_index();
  drone_trajectory_port_ = DeclareVectorInputPort("drone_trajectory", 9).get_index();
  desired_tension_port_ = DeclareVectorInputPort("desired_tension", 1).get_index();
  desired_cable_direction_port_ = DeclareVectorInputPort("desired_cable_direction", 3).get_index();
  actual_cable_direction_port_ = DeclareVectorInputPort("actual_cable_direction", 3).get_index();
  actual_tension_port_ = DeclareVectorInputPort("actual_tension", 1).get_index();
  cable_angular_velocity_port_ = DeclareVectorInputPort("cable_angular_velocity", 3).get_index();
  disturbance_port_ = DeclareVectorInputPort("disturbance_estimate", 3).get_index();

  // Continuous state for integral
  DeclareContinuousState(3);

  // === Output ports ===

  thrust_force_port_ = DeclareVectorOutputPort(
      "thrust_force", 3,
      &GPACLoadTrackingController::CalcThrustForce).get_index();

  thrust_magnitude_port_ = DeclareVectorOutputPort(
      "thrust_magnitude", 1,
      &GPACLoadTrackingController::CalcThrustMagnitude).get_index();

  desired_attitude_port_ = DeclareVectorOutputPort(
      "desired_attitude", 4,
      &GPACLoadTrackingController::CalcDesiredAttitude).get_index();

  desired_angular_velocity_port_ = DeclareVectorOutputPort(
      "desired_angular_velocity", 3,
      &GPACLoadTrackingController::CalcDesiredAngularVelocity).get_index();

  antiswing_force_port_ = DeclareVectorOutputPort(
      "antiswing_force", 3,
      &GPACLoadTrackingController::CalcAntiSwingForce).get_index();

  cable_error_port_ = DeclareVectorOutputPort(
      "cable_error", 3,
      &GPACLoadTrackingController::CalcCableError).get_index();
}

void GPACLoadTrackingController::SetDefaultState(
    const Context<double>& context,
    drake::systems::State<double>* state) const {
  state->get_mutable_continuous_state()
      .get_mutable_vector()
      .SetFromVector(Eigen::Vector3d::Zero());
}

void GPACLoadTrackingController::DoCalcTimeDerivatives(
    const Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {

  // Parse drone state
  const Eigen::VectorXd& state = get_drone_state_input().Eval(context);
  Eigen::Vector3d p = state.segment<3>(0);

  // Parse trajectory
  const Eigen::VectorXd& traj = get_drone_trajectory_input().Eval(context);
  Eigen::Vector3d p_des = traj.segment<3>(0);

  // Position error
  Eigen::Vector3d e = p - p_des;

  // Anti-windup
  const Eigen::Vector3d& integral =
      context.get_continuous_state().get_vector().CopyToVector();

  Eigen::Vector3d integral_dot = e;

  for (int i = 0; i < 3; ++i) {
    if (std::abs(integral[i]) > params_.max_integral &&
        e[i] * integral[i] > 0) {
      integral_dot[i] = 0.0;
    }
  }

  derivatives->get_mutable_vector().SetFromVector(integral_dot);
}

Eigen::Vector3d GPACLoadTrackingController::ComputeAntiSwingForce(
    const Context<double>& context) const {

  if (!params_.enable_antiswing) {
    return Eigen::Vector3d::Zero();
  }

  // Get cable directions
  Eigen::Vector3d q_des = get_desired_cable_direction_input().Eval(context);
  Eigen::Vector3d q = get_actual_cable_direction_input().Eval(context);

  // Normalize
  if (q_des.norm() > 1e-6) q_des.normalize();
  else q_des = Eigen::Vector3d(0, 0, -1);

  if (q.norm() > 1e-6) q.normalize();
  else q = Eigen::Vector3d(0, 0, -1);

  // Cable direction error [Eq. 8]: e_q = P(q)*q_des
  // This lies in the tangent space T_q S²
  Eigen::Vector3d e_q = CableDirectionError(q, q_des);

  // Cable angular velocity damping
  Eigen::Vector3d omega_q = Eigen::Vector3d::Zero();
  const auto& omega_port = get_input_port(cable_angular_velocity_port_);
  if (omega_port.HasValue(context)) {
    omega_q = omega_port.Eval(context);
    // Project to tangent space (perpendicular to q)
    omega_q = ProjectToTangentS2(q, omega_q);
  }

  // Anti-swing control law:
  // F_antiswing = -k_q * e_q - k_w * ω_q
  //
  // This generates a restoring force that tries to align q with q_des
  // and damps cable oscillations.
  //
  // The negative sign is because e_q points from q toward q_des,
  // and we want to push the cable in that direction.

  Eigen::Vector3d F_antiswing = params_.kq * e_q - params_.kw * omega_q;

  return F_antiswing;
}

Eigen::Vector3d GPACLoadTrackingController::ComputeThrustVector(
    const Context<double>& context) const {

  // === Parse inputs ===
  const Eigen::VectorXd& state = get_drone_state_input().Eval(context);
  Eigen::Vector3d p = state.segment<3>(0);
  Eigen::Vector3d v = state.segment<3>(7);

  const Eigen::VectorXd& traj = get_drone_trajectory_input().Eval(context);
  Eigen::Vector3d p_des = traj.segment<3>(0);
  Eigen::Vector3d v_des = traj.segment<3>(3);
  Eigen::Vector3d a_des = traj.segment<3>(6);

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
  Eigen::Vector3d e = p - p_des;
  Eigen::Vector3d e_dot = v - v_des;

  Eigen::Vector3d e_int =
      context.get_continuous_state().get_vector().CopyToVector();

  // Gain matrices
  Eigen::Matrix3d Kp = params_.Kp.asDiagonal();
  Eigen::Matrix3d Kd = params_.Kd.asDiagonal();
  Eigen::Matrix3d Ki = params_.Ki.asDiagonal();

  // Gravity
  Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // === Feedforward: gravity + trajectory ===
  Eigen::Vector3d f_feedforward = params_.mass * (a_des + params_.gravity * e3);

  // === Cable tension compensation ===
  Eigen::Vector3d f_cable;
  if (params_.use_actual_tension_feedforward) {
    f_cable = params_.cable_compensation_gain * T_actual * q_actual;
  } else {
    f_cable = params_.cable_compensation_gain * T_des * q_des;
  }

  // === Feedback: PID ===
  Eigen::Vector3d f_feedback = -Kp * e - Kd * e_dot - Ki * e_int;

  // === Anti-swing control (S²) ===
  Eigen::Vector3d f_antiswing = ComputeAntiSwingForce(context);

  // === ESO disturbance feedforward ===
  Eigen::Vector3d f_eso = Eigen::Vector3d::Zero();
  if (params_.enable_eso_feedforward) {
    const auto& dist_port = get_input_port(disturbance_port_);
    if (dist_port.HasValue(context)) {
      // Disturbance is in acceleration units, multiply by mass
      f_eso = params_.mass * dist_port.Eval(context);
    }
  }

  // === Total thrust ===
  Eigen::Vector3d f_total = f_feedforward + f_cable + f_feedback + f_antiswing + f_eso;

  // === Saturation ===
  double thrust_norm = f_total.norm();
  if (thrust_norm > params_.max_thrust) {
    f_total = f_total * (params_.max_thrust / thrust_norm);
  }
  if (thrust_norm < params_.min_thrust) {
    f_total = params_.min_thrust * e3;
  }

  return f_total;
}

Eigen::Quaterniond GPACLoadTrackingController::ThrustVectorToQuaternion(
    const Eigen::Vector3d& thrust, double yaw_des) const {

  Eigen::Vector3d z_body = thrust.normalized();
  Eigen::Vector3d x_c(std::cos(yaw_des), std::sin(yaw_des), 0.0);

  // Handle near-vertical thrust
  if (std::abs(z_body.dot(Eigen::Vector3d::UnitZ())) > 0.99) {
    // Near vertical - use yaw to define x-axis
    Eigen::Vector3d y_body = z_body.cross(x_c);
    if (y_body.norm() < 1e-6) {
      y_body = Eigen::Vector3d(0, 1, 0);
    }
    y_body.normalize();
    Eigen::Vector3d x_body = y_body.cross(z_body);

    Eigen::Matrix3d R;
    R.col(0) = x_body;
    R.col(1) = y_body;
    R.col(2) = z_body;

    return Eigen::Quaterniond(R).normalized();
  }

  Eigen::Vector3d y_body = z_body.cross(x_c).normalized();
  Eigen::Vector3d x_body = y_body.cross(z_body);

  Eigen::Matrix3d R;
  R.col(0) = x_body;
  R.col(1) = y_body;
  R.col(2) = z_body;

  Eigen::Quaterniond q(R);
  if (q.w() < 0) {
    q.coeffs() = -q.coeffs();
  }
  return q.normalized();
}

void GPACLoadTrackingController::CalcThrustForce(
    const Context<double>& context,
    BasicVector<double>* output) const {
  output->SetFromVector(ComputeThrustVector(context));
}

void GPACLoadTrackingController::CalcThrustMagnitude(
    const Context<double>& context,
    BasicVector<double>* output) const {
  output->SetAtIndex(0, ComputeThrustVector(context).norm());
}

void GPACLoadTrackingController::CalcDesiredAttitude(
    const Context<double>& context,
    BasicVector<double>* output) const {
  Eigen::Vector3d thrust = ComputeThrustVector(context);
  Eigen::Quaterniond q = ThrustVectorToQuaternion(thrust, 0.0);

  output->SetAtIndex(0, q.w());
  output->SetAtIndex(1, q.x());
  output->SetAtIndex(2, q.y());
  output->SetAtIndex(3, q.z());
}

void GPACLoadTrackingController::CalcDesiredAngularVelocity(
    const Context<double>& context,
    BasicVector<double>* output) const {
  // For now, return zero (feedforward requires trajectory derivatives)
  output->SetFromVector(Eigen::Vector3d::Zero());
}

void GPACLoadTrackingController::CalcAntiSwingForce(
    const Context<double>& context,
    BasicVector<double>* output) const {
  output->SetFromVector(ComputeAntiSwingForce(context));
}

void GPACLoadTrackingController::CalcCableError(
    const Context<double>& context,
    BasicVector<double>* output) const {

  Eigen::Vector3d q_des = get_desired_cable_direction_input().Eval(context);
  Eigen::Vector3d q = get_actual_cable_direction_input().Eval(context);

  if (q_des.norm() > 1e-6) q_des.normalize();
  else q_des = Eigen::Vector3d(0, 0, -1);

  if (q.norm() > 1e-6) q.normalize();
  else q = Eigen::Vector3d(0, 0, -1);

  Eigen::Vector3d e_q = CableDirectionError(q, q_des);
  output->SetFromVector(e_q);
}

}  // namespace tether_lift
