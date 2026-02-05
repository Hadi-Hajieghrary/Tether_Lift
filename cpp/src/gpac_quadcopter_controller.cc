/// @file gpac_quadcopter_controller.cc
/// @brief GPAC-enhanced quadcopter controller implementation.

#include "gpac_quadcopter_controller.h"

#include <algorithm>
#include <cmath>

#include <drake/multibody/math/spatial_force.h>

namespace quad_rope_lift {

using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::SpatialForce;
using drake::systems::BasicVector;
using drake::systems::Context;

GPACQuadcopterController::GPACQuadcopterController(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::multibody::RigidBody<double>& quadcopter_body,
    const GPACParams& params)
    : plant_(plant),
      quad_body_index_(quadcopter_body.index()),
      params_(params),
      internal_eso_(params.eso_omega, params.eso_b0) {

  // Precompute ascent trajectory
  const double altitude_change = params_.final_altitude - params_.initial_altitude;
  ascent_duration_ = (params_.climb_rate > 0.0) ?
      std::abs(altitude_change) / params_.climb_rate : 0.0;
  ascent_direction_ = (altitude_change >= 0.0) ? 1.0 : -1.0;

  // === Input ports ===

  plant_state_port_ = DeclareVectorInputPort(
      "plant_state",
      BasicVector<double>(plant.num_positions() + plant.num_velocities()))
      .get_index();

  tension_port_ = DeclareVectorInputPort(
      "rope_tension", BasicVector<double>(4))
      .get_index();

  cable_direction_port_ = DeclareVectorInputPort(
      "cable_direction", BasicVector<double>(3))
      .get_index();

  estimated_state_port_ = DeclareVectorInputPort(
      "estimated_state", BasicVector<double>(6))
      .get_index();

  disturbance_port_ = DeclareVectorInputPort(
      "disturbance_estimate", BasicVector<double>(3))
      .get_index();

  // === Continuous state for integral control ===
  DeclareContinuousState(3);  // [integral_x, integral_y, integral_z]

  // === Output ports ===

  control_port_ = DeclareAbstractOutputPort(
      "control_force",
      &GPACQuadcopterController::CalcControlForce)
      .get_index();

  control_vector_port_ = DeclareVectorOutputPort(
      "control_vector", 6,
      &GPACQuadcopterController::CalcControlVector)
      .get_index();

  desired_attitude_port_ = DeclareVectorOutputPort(
      "desired_attitude", 4,
      &GPACQuadcopterController::CalcDesiredAttitude)
      .get_index();

  attitude_error_port_ = DeclareVectorOutputPort(
      "attitude_error", 3,
      &GPACQuadcopterController::CalcAttitudeError)
      .get_index();
}

void GPACQuadcopterController::DoCalcTimeDerivatives(
    const drake::systems::Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {

  // Get state vectors
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);

  // Create plant context
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Get position and velocity
  const auto& quad_body = plant_.get_body(quad_body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, quad_body);
  const auto& velocity_world = plant_.EvalBodySpatialVelocityInWorld(*plant_context, quad_body);

  Eigen::Vector3d position = pose_world.translation();
  Eigen::Vector3d velocity = velocity_world.translational();

  // Override with estimated state if connected
  const auto& est_port = get_input_port(estimated_state_port_);
  if (est_port.HasValue(context)) {
    const auto& estimated = est_port.Eval(context);
    position = Eigen::Vector3d(estimated[0], estimated[1], estimated[2]);
    velocity = Eigen::Vector3d(estimated[3], estimated[4], estimated[5]);
  }

  // Compute desired trajectory
  const double t = context.get_time();
  Eigen::Vector3d pos_des, vel_des, acc_des;
  ComputeTrajectory(t, pos_des, vel_des, acc_des);

  // Position error for integral
  const Eigen::Vector3d pos_error = pos_des - position;

  // Integral derivative = error (with anti-windup check in control)
  auto& deriv = derivatives->get_mutable_vector();
  deriv.SetAtIndex(0, pos_error.x());
  deriv.SetAtIndex(1, pos_error.y());
  deriv.SetAtIndex(2, pos_error.z());
}

void GPACQuadcopterController::SetDefaultState(
    const drake::systems::Context<double>& context,
    drake::systems::State<double>* state) const {
  auto& continuous_state = state->get_mutable_continuous_state();
  continuous_state.get_mutable_vector().SetZero();
}

void GPACQuadcopterController::ComputeTrajectory(
    double t,
    Eigen::Vector3d& pos_des,
    Eigen::Vector3d& vel_des,
    Eigen::Vector3d& acc_des) const {

  acc_des.setZero();

  if (!params_.waypoints.empty()) {
    // Waypoint-based trajectory
    double segment_start_time = 0.0;
    for (size_t i = 0; i < params_.waypoints.size(); ++i) {
      const auto& wp = params_.waypoints[i];
      const double segment_end_time = wp.arrival_time;
      const double hold_end_time = segment_end_time + wp.hold_time;

      if (t <= segment_end_time) {
        if (i == 0) {
          pos_des = wp.position + params_.formation_offset;
          vel_des.setZero();
        } else {
          const auto& prev_wp = params_.waypoints[i - 1];
          const double segment_duration = segment_end_time - segment_start_time;
          if (segment_duration > 1e-6) {
            const double alpha = (t - segment_start_time) / segment_duration;
            pos_des = (1.0 - alpha) * prev_wp.position + alpha * wp.position + params_.formation_offset;
            vel_des = (wp.position - prev_wp.position) / segment_duration;
          } else {
            pos_des = wp.position + params_.formation_offset;
            vel_des.setZero();
          }
        }
        return;
      } else if (t <= hold_end_time) {
        pos_des = wp.position + params_.formation_offset;
        vel_des.setZero();
        return;
      }
      segment_start_time = hold_end_time;
    }
    pos_des = params_.waypoints.back().position + params_.formation_offset;
    vel_des.setZero();
  } else {
    // Legacy altitude trajectory
    double desired_altitude;
    double desired_velocity_z;

    if (t <= params_.ascent_start_time || params_.climb_rate <= 0.0) {
      desired_altitude = params_.initial_altitude;
      desired_velocity_z = 0.0;
    } else if (t >= params_.ascent_start_time + ascent_duration_) {
      desired_altitude = params_.final_altitude;
      desired_velocity_z = 0.0;
    } else {
      const double elapsed = t - params_.ascent_start_time;
      desired_altitude = params_.initial_altitude + ascent_direction_ * params_.climb_rate * elapsed;
      desired_velocity_z = ascent_direction_ * params_.climb_rate;
    }

    pos_des = Eigen::Vector3d(params_.formation_offset.x(),
                              params_.formation_offset.y(),
                              desired_altitude);
    vel_des = Eigen::Vector3d(0.0, 0.0, desired_velocity_z);
  }
}

Eigen::Vector3d GPACQuadcopterController::ComputeLayer1Control(
    const Context<double>& context,
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    double measured_tension) const {

  const double t = context.get_time();

  // Get integral state (with anti-windup)
  const auto& state = context.get_continuous_state_vector();
  Eigen::Vector3d pos_integral(
      std::clamp(state[0], -params_.max_integral, params_.max_integral),
      std::clamp(state[1], -params_.max_integral, params_.max_integral),
      std::clamp(state[2], -params_.max_integral, params_.max_integral));

  // Compute trajectory
  Eigen::Vector3d pos_des, vel_des, acc_des;
  ComputeTrajectory(t, pos_des, vel_des, acc_des);

  // Position and velocity errors
  const Eigen::Vector3d pos_error = pos_des - position;
  const Eigen::Vector3d vel_error = vel_des - velocity;

  // === Pickup phase logic ===
  bool in_pickup_phase = false;
  double target_tension = params_.pickup_target_tension;

  if (!pickup_start_time_.has_value() && measured_tension >= params_.pickup_detection_threshold) {
    pickup_start_time_ = t;
  }

  if (pickup_start_time_.has_value()) {
    const double time_since_pickup = t - pickup_start_time_.value();
    if (time_since_pickup >= 0.0 && time_since_pickup <= params_.pickup_ramp_duration) {
      in_pickup_phase = true;
      double ramp_fraction = std::clamp(time_since_pickup / params_.pickup_ramp_duration, 0.0, 1.0);
      target_tension = ramp_fraction * params_.pickup_target_tension;

      const double tension_error = target_tension - measured_tension;
      double altitude_adjustment = params_.tension_altitude_gain * tension_error;
      altitude_adjustment = std::clamp(altitude_adjustment,
                                       -params_.tension_altitude_max,
                                       params_.tension_altitude_max);
      pos_des.z() += altitude_adjustment;
    }
  }

  // === Anti-swing control (S² cable direction) ===
  Eigen::Vector3d anti_swing_term = Eigen::Vector3d::Zero();
  if (params_.enable_antiswing) {
    const auto& cable_port = get_input_port(cable_direction_port_);
    if (cable_port.HasValue(context)) {
      const Eigen::Vector3d q = cable_port.Eval(context).normalized();

      // Desired cable direction: straight down
      const Eigen::Vector3d q_d(0.0, 0.0, -1.0);

      // Cable direction error [Eq. 8]: e_q = P(q)*q_d
      const Eigen::Vector3d e_q = gpac::CableDirectionError(q, q_d);

      // Anti-swing control (simplified - no cable velocity measurement)
      anti_swing_term = params_.cable_kq * e_q;
    }
  }

  // === Disturbance feedforward ===
  Eigen::Vector3d disturbance_ff = Eigen::Vector3d::Zero();
  if (params_.enable_eso_feedforward) {
    const auto& dist_port = get_input_port(disturbance_port_);
    if (dist_port.HasValue(context)) {
      disturbance_ff = dist_port.Eval(context);
    } else {
      // Use internal ESO
      const double dt = (last_update_time_ > 0.0) ? (t - last_update_time_) : 0.001;
      if (dt > 0.0001 && last_update_time_ != t) {
        internal_eso_.Update(position, last_control_accel_, dt);
        last_update_time_ = t;
      }
      disturbance_ff = internal_eso_.disturbance();
    }
  }

  // === Build control force ===
  const Eigen::Vector3d e3(0.0, 0.0, 1.0);

  // Feedforward: gravity + trajectory acceleration
  Eigen::Vector3d F_ff = params_.mass * (params_.gravity * e3 + acc_des);

  // Feedback: PID position control
  Eigen::Vector3d F_fb = params_.mass * (
      params_.position_kp * pos_error +
      params_.position_kd * vel_error +
      params_.position_ki * pos_integral);

  // Cable tension feedforward
  double tension_ff_limit = in_pickup_phase ? target_tension : params_.pickup_target_tension;
  double tension_ff = std::clamp(measured_tension, 0.0, tension_ff_limit);
  Eigen::Vector3d F_tension = tension_ff * e3;

  // Tension feedback (during pickup)
  if (in_pickup_phase) {
    F_tension += params_.tension_kp * (target_tension - measured_tension) * e3;
  }

  // ESO disturbance compensation
  Eigen::Vector3d F_eso = params_.mass * disturbance_ff;

  // Total desired thrust vector
  Eigen::Vector3d F_des = F_ff + F_fb + F_tension + F_eso + anti_swing_term;

  // Store for ESO
  last_control_accel_ = F_des / params_.mass;

  return F_des;
}

Eigen::Vector3d GPACQuadcopterController::ComputeLayer2Control(
    const Eigen::Matrix3d& R,
    const Eigen::Matrix3d& R_d,
    const Eigen::Vector3d& Omega,
    const Eigen::Vector3d& Omega_d) const {

  // Attitude error [Eq. 12]: e_R = ½(R_d^T R - R^T R_d)^∨
  const Eigen::Vector3d e_R = gpac::AttitudeError(R, R_d);

  // Angular velocity error [Eq. 12]: e_Ω = Ω - R^T R_d Ω_d
  const Eigen::Vector3d e_Omega = gpac::AngularVelocityError(Omega, R, R_d, Omega_d);

  // Geometric control law [Eq. 14]:
  // τ = -k_R e_R - k_Ω e_Ω + Ω × J Ω - J(Ω̂ R^T R_d Ω_d - R^T R_d Ω̇_d)
  //
  // Simplified (assuming Ω_d ≈ 0, ignoring feedforward):
  // τ = -k_R e_R - k_Ω e_Ω

  Eigen::Vector3d tau = -params_.attitude_kR * e_R - params_.attitude_kOmega * e_Omega;

  // Saturate torque
  for (int i = 0; i < 3; ++i) {
    tau[i] = std::clamp(tau[i], -params_.max_torque, params_.max_torque);
  }

  return tau;
}

void GPACQuadcopterController::CalcControlForce(
    const Context<double>& context,
    std::vector<ExternallyAppliedSpatialForce<double>>* output) const {

  // Read inputs
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);
  const auto& tension_data = get_input_port(tension_port_).Eval(context);
  const double measured_tension = tension_data[0];

  // Create plant context
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // Get pose and velocity
  const auto& quad_body = plant_.get_body(quad_body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, quad_body);
  const auto& velocity_world = plant_.EvalBodySpatialVelocityInWorld(*plant_context, quad_body);

  // Position and velocity (override with estimated state if available)
  Eigen::Vector3d position = pose_world.translation();
  Eigen::Vector3d velocity = velocity_world.translational();

  const auto& est_port = get_input_port(estimated_state_port_);
  if (est_port.HasValue(context)) {
    const auto& estimated = est_port.Eval(context);
    position = Eigen::Vector3d(estimated[0], estimated[1], estimated[2]);
    velocity = Eigen::Vector3d(estimated[3], estimated[4], estimated[5]);
  }

  // Current attitude
  const Eigen::Matrix3d R = pose_world.rotation().matrix();

  // Angular velocity in body frame
  const Eigen::Vector3d omega_W = velocity_world.rotational();
  const Eigen::Vector3d Omega = R.transpose() * omega_W;

  // === Layer 1: Position + Anti-swing Control ===
  const Eigen::Vector3d F_des = ComputeLayer1Control(context, position, velocity, measured_tension);

  // Thrust magnitude
  double thrust = F_des.norm();
  thrust = std::clamp(thrust, params_.min_thrust, params_.max_thrust);

  // === Layer 2: Geometric Attitude Control ===
  // Desired rotation from thrust direction [Eq. 19-20]
  const double psi_d = 0.0;  // Keep yaw at zero
  const Eigen::Matrix3d R_d = gpac::DesiredRotation(F_des, psi_d);

  // Desired angular velocity (zero for now - could add feedforward)
  const Eigen::Vector3d Omega_d = Eigen::Vector3d::Zero();

  // Compute control torque
  const Eigen::Vector3d tau_body = ComputeLayer2Control(R, R_d, Omega, Omega_d);

  // Transform torque to world frame
  const Eigen::Vector3d torque_world = R * tau_body;

  // Thrust in body z-direction, transformed to world frame
  const Eigen::Vector3d force_world = R.col(2) * thrust;

  // Package output
  output->clear();
  output->reserve(1);

  ExternallyAppliedSpatialForce<double> control_wrench;
  control_wrench.body_index = quad_body_index_;
  control_wrench.p_BoBq_B = Eigen::Vector3d::Zero();
  control_wrench.F_Bq_W = SpatialForce<double>(torque_world, force_world);
  output->push_back(control_wrench);
}

void GPACQuadcopterController::CalcControlVector(
    const Context<double>& context,
    BasicVector<double>* output) const {

  std::vector<ExternallyAppliedSpatialForce<double>> forces;
  CalcControlForce(context, &forces);

  if (!forces.empty()) {
    const auto& F = forces[0].F_Bq_W;
    output->SetAtIndex(0, F.rotational()(0));
    output->SetAtIndex(1, F.rotational()(1));
    output->SetAtIndex(2, F.rotational()(2));
    output->SetAtIndex(3, F.translational()(0));
    output->SetAtIndex(4, F.translational()(1));
    output->SetAtIndex(5, F.translational()(2));
  } else {
    for (int i = 0; i < 6; ++i) {
      output->SetAtIndex(i, 0.0);
    }
  }
}

void GPACQuadcopterController::CalcDesiredAttitude(
    const Context<double>& context,
    BasicVector<double>* output) const {

  // Read inputs
  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);
  const auto& tension_data = get_input_port(tension_port_).Eval(context);
  const double measured_tension = tension_data[0];

  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  const auto& quad_body = plant_.get_body(quad_body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, quad_body);
  const auto& velocity_world = plant_.EvalBodySpatialVelocityInWorld(*plant_context, quad_body);

  Eigen::Vector3d position = pose_world.translation();
  Eigen::Vector3d velocity = velocity_world.translational();

  const auto& est_port = get_input_port(estimated_state_port_);
  if (est_port.HasValue(context)) {
    const auto& estimated = est_port.Eval(context);
    position = Eigen::Vector3d(estimated[0], estimated[1], estimated[2]);
    velocity = Eigen::Vector3d(estimated[3], estimated[4], estimated[5]);
  }

  const Eigen::Vector3d F_des = ComputeLayer1Control(context, position, velocity, measured_tension);
  const Eigen::Matrix3d R_d = gpac::DesiredRotation(F_des, 0.0);
  const Eigen::Quaterniond q_d = gpac::RotationToQuaternion(R_d);

  output->SetAtIndex(0, q_d.w());
  output->SetAtIndex(1, q_d.x());
  output->SetAtIndex(2, q_d.y());
  output->SetAtIndex(3, q_d.z());
}

void GPACQuadcopterController::CalcAttitudeError(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state_vector = get_input_port(plant_state_port_).Eval(context);
  const auto& tension_data = get_input_port(tension_port_).Eval(context);
  const double measured_tension = tension_data[0];

  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  const auto& quad_body = plant_.get_body(quad_body_index_);
  const auto& pose_world = plant_.EvalBodyPoseInWorld(*plant_context, quad_body);
  const auto& velocity_world = plant_.EvalBodySpatialVelocityInWorld(*plant_context, quad_body);

  Eigen::Vector3d position = pose_world.translation();
  Eigen::Vector3d velocity = velocity_world.translational();

  const auto& est_port = get_input_port(estimated_state_port_);
  if (est_port.HasValue(context)) {
    const auto& estimated = est_port.Eval(context);
    position = Eigen::Vector3d(estimated[0], estimated[1], estimated[2]);
    velocity = Eigen::Vector3d(estimated[3], estimated[4], estimated[5]);
  }

  const Eigen::Matrix3d R = pose_world.rotation().matrix();
  const Eigen::Vector3d F_des = ComputeLayer1Control(context, position, velocity, measured_tension);
  const Eigen::Matrix3d R_d = gpac::DesiredRotation(F_des, 0.0);

  const Eigen::Vector3d e_R = gpac::AttitudeError(R, R_d);

  output->SetAtIndex(0, e_R.x());
  output->SetAtIndex(1, e_R.y());
  output->SetAtIndex(2, e_R.z());
}

}  // namespace quad_rope_lift
