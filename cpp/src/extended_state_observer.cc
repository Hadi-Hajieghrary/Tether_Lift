/// @file extended_state_observer.cc
/// @brief Implementation of Extended State Observer for disturbance estimation.

#include "extended_state_observer.h"

#include <cmath>
#include <algorithm>

namespace quad_rope_lift {
namespace gpac {

// =============================================================================
// AxisESO Implementation
// =============================================================================

AxisESO::AxisESO(double omega_o, double b0)
    : z_({0.0, 0.0, 0.0}),
      omega_o_(omega_o),
      b0_(b0),
      error_(0.0) {
  ComputeGains();
}

void AxisESO::Reset(double initial_position, double initial_velocity) {
  z_[0] = initial_position;
  z_[1] = initial_velocity;
  z_[2] = 0.0;
  error_ = 0.0;
}

void AxisESO::Update(double y, double u, double dt) {
  error_ = y - z_[0];

  // ESO dynamics
  const double z1_dot = z_[1] + l1_ * error_;
  const double z2_dot = z_[2] + l2_ * error_ + b0_ * u;
  const double z3_dot = l3_ * error_;

  // Euler integration
  z_[0] += z1_dot * dt;
  z_[1] += z2_dot * dt;
  z_[2] += z3_dot * dt;

  // Saturate disturbance estimate
  z_[2] = std::max(-max_disturbance_, std::min(max_disturbance_, z_[2]));
}

void AxisESO::set_omega(double omega_o) {
  omega_o_ = omega_o;
  ComputeGains();
}

void AxisESO::ComputeGains() {
  // Triple pole at -ω_o
  l1_ = 3.0 * omega_o_;
  l2_ = 3.0 * omega_o_ * omega_o_;
  l3_ = omega_o_ * omega_o_ * omega_o_;
}

// =============================================================================
// DroneESO Implementation
// =============================================================================

DroneESO::DroneESO(double omega_o, double b0)
    : axes_({AxisESO(omega_o, b0),
             AxisESO(omega_o, b0),
             AxisESO(omega_o, b0)}) {}

void DroneESO::Reset(const Eigen::Vector3d& initial_position,
                     const Eigen::Vector3d& initial_velocity) {
  for (int i = 0; i < 3; ++i) {
    axes_[i].Reset(initial_position[i], initial_velocity[i]);
  }
}

void DroneESO::Update(const Eigen::Vector3d& position_measurement,
                      const Eigen::Vector3d& control_input,
                      double dt) {
  for (int i = 0; i < 3; ++i) {
    axes_[i].Update(position_measurement[i], control_input[i], dt);
  }
}

Eigen::Vector3d DroneESO::position() const {
  return Eigen::Vector3d(axes_[0].position(),
                         axes_[1].position(),
                         axes_[2].position());
}

Eigen::Vector3d DroneESO::velocity() const {
  return Eigen::Vector3d(axes_[0].velocity(),
                         axes_[1].velocity(),
                         axes_[2].velocity());
}

Eigen::Vector3d DroneESO::disturbance() const {
  return Eigen::Vector3d(axes_[0].disturbance(),
                         axes_[1].disturbance(),
                         axes_[2].disturbance());
}

Eigen::Vector3d DroneESO::error() const {
  return Eigen::Vector3d(axes_[0].error(),
                         axes_[1].error(),
                         axes_[2].error());
}

void DroneESO::set_omega(double omega_o) {
  for (auto& axis : axes_) {
    axis.set_omega(omega_o);
  }
}

void DroneESO::set_b0(double b0) {
  for (auto& axis : axes_) {
    axis.set_b0(b0);
  }
}

// =============================================================================
// ExtendedStateObserver (Drake LeafSystem) Implementation
// =============================================================================

ExtendedStateObserver::ExtendedStateObserver(const ESOParams& params)
    : params_(params) {

  // Input ports
  position_port_ = DeclareVectorInputPort("position_measurement", 3).get_index();
  control_port_ = DeclareVectorInputPort("control_input", 3).get_index();

  // Continuous state: 9 states [z1_x, z2_x, z3_x, z1_y, z2_y, z3_y, z1_z, z2_z, z3_z]
  DeclareContinuousState(9);

  // Output ports
  est_position_port_ = DeclareVectorOutputPort(
      "estimated_position", 3,
      &ExtendedStateObserver::CalcEstimatedPosition).get_index();

  est_velocity_port_ = DeclareVectorOutputPort(
      "estimated_velocity", 3,
      &ExtendedStateObserver::CalcEstimatedVelocity).get_index();

  est_disturbance_port_ = DeclareVectorOutputPort(
      "estimated_disturbance", 3,
      &ExtendedStateObserver::CalcEstimatedDisturbance).get_index();
}

void ExtendedStateObserver::DoCalcTimeDerivatives(
    const drake::systems::Context<double>& context,
    drake::systems::ContinuousState<double>* derivatives) const {

  const Eigen::Vector3d y = get_input_port(position_port_).Eval(context);
  const Eigen::Vector3d u = get_input_port(control_port_).Eval(context);

  const auto& state = context.get_continuous_state_vector();

  const double omega = params_.omega_o;
  const double l1 = 3.0 * omega;
  const double l2 = 3.0 * omega * omega;
  const double l3 = omega * omega * omega;
  const double b0 = params_.b0;

  auto& deriv = derivatives->get_mutable_vector();

  for (int axis = 0; axis < 3; ++axis) {
    const int base = axis * 3;

    const double z1 = state[base + 0];
    const double z2 = state[base + 1];
    const double z3 = state[base + 2];

    const double error = y[axis] - z1;

    deriv[base + 0] = z2 + l1 * error;
    deriv[base + 1] = z3 + l2 * error + b0 * u[axis];
    deriv[base + 2] = l3 * error;
  }
}

void ExtendedStateObserver::SetDefaultState(
    const drake::systems::Context<double>& context,
    drake::systems::State<double>* state) const {
  auto& continuous_state = state->get_mutable_continuous_state();
  continuous_state.get_mutable_vector().SetZero();
}

void ExtendedStateObserver::CalcEstimatedPosition(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  const auto& state = context.get_continuous_state_vector();
  output->get_mutable_value()[0] = state[0];
  output->get_mutable_value()[1] = state[3];
  output->get_mutable_value()[2] = state[6];
}

void ExtendedStateObserver::CalcEstimatedVelocity(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  const auto& state = context.get_continuous_state_vector();
  output->get_mutable_value()[0] = state[1];
  output->get_mutable_value()[1] = state[4];
  output->get_mutable_value()[2] = state[7];
}

void ExtendedStateObserver::CalcEstimatedDisturbance(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  const auto& state = context.get_continuous_state_vector();
  const double max_d = params_.max_disturbance;

  for (int axis = 0; axis < 3; ++axis) {
    const double z3 = state[axis * 3 + 2];
    output->get_mutable_value()[axis] = std::max(-max_d, std::min(max_d, z3));
  }
}

}  // namespace gpac
}  // namespace quad_rope_lift
