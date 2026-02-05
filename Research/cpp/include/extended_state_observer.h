#pragma once

/// @file extended_state_observer.h
/// @brief Extended State Observer for disturbance estimation [GPAC Layer 4]
///
/// Implements a third-order ESO for each spatial axis:
///
///   ż₁ = z₂ + l₁(y - z₁)
///   ż₂ = z₃ + l₂(y - z₁) + b₀u
///   ż₃ = l₃(y - z₁)
///
/// where:
///   - y is the measured position
///   - u is the control input (force/mass)
///   - z₁ estimates position
///   - z₂ estimates velocity
///   - z₃ estimates the total disturbance
///
/// Gain design (triple pole at -ω_o):
///   l₁ = 3ω_o, l₂ = 3ω_o², l₃ = ω_o³

#include <drake/systems/framework/leaf_system.h>
#include <Eigen/Core>
#include <array>

namespace quad_rope_lift {
namespace gpac {

/// @brief Parameters for the Extended State Observer
struct ESOParams {
  double omega_o = 50.0;        ///< Observer bandwidth [rad/s]
  double b0 = 1.0 / 1.5;        ///< Input gain (1/mass)
  double max_disturbance = 20.0; ///< Saturation limit [m/s²]
};

/// @brief Single-axis Extended State Observer
class AxisESO {
 public:
  AxisESO(double omega_o = 50.0, double b0 = 1.0 / 1.5);

  void Reset(double initial_position, double initial_velocity = 0.0);
  void Update(double y, double u, double dt);

  double position() const { return z_[0]; }
  double velocity() const { return z_[1]; }
  double disturbance() const { return z_[2]; }
  double error() const { return error_; }

  void set_omega(double omega_o);
  void set_b0(double b0) { b0_ = b0; }
  void set_max_disturbance(double max_d) { max_disturbance_ = max_d; }

 private:
  std::array<double, 3> z_;
  double l1_, l2_, l3_;
  double omega_o_;
  double b0_;
  double error_;
  double max_disturbance_ = 20.0;

  void ComputeGains();
};

/// @brief Three-axis Extended State Observer (Drake LeafSystem)
class ExtendedStateObserver final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedStateObserver);

  explicit ExtendedStateObserver(const ESOParams& params = ESOParams());

  // Input port accessors
  const drake::systems::InputPort<double>& get_position_input_port() const {
    return get_input_port(position_port_);
  }

  const drake::systems::InputPort<double>& get_control_input_port() const {
    return get_input_port(control_port_);
  }

  // Output port accessors
  const drake::systems::OutputPort<double>& get_estimated_position_output_port() const {
    return get_output_port(est_position_port_);
  }

  const drake::systems::OutputPort<double>& get_estimated_velocity_output_port() const {
    return get_output_port(est_velocity_port_);
  }

  const drake::systems::OutputPort<double>& get_estimated_disturbance_output_port() const {
    return get_output_port(est_disturbance_port_);
  }

  const ESOParams& params() const { return params_; }
  void set_omega(double omega_o) { params_.omega_o = omega_o; }
  void set_b0(double b0) { params_.b0 = b0; }

 private:
  void DoCalcTimeDerivatives(
      const drake::systems::Context<double>& context,
      drake::systems::ContinuousState<double>* derivatives) const override;

  void SetDefaultState(
      const drake::systems::Context<double>& context,
      drake::systems::State<double>* state) const override;

  void CalcEstimatedPosition(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcEstimatedVelocity(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcEstimatedDisturbance(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  ESOParams params_;

  int position_port_{};
  int control_port_{};
  int est_position_port_{};
  int est_velocity_port_{};
  int est_disturbance_port_{};
};

/// @brief Per-drone ESO wrapper for convenient standalone usage
class DroneESO {
 public:
  DroneESO(double omega_o = 50.0, double b0 = 1.0 / 1.5);

  void Reset(const Eigen::Vector3d& initial_position,
             const Eigen::Vector3d& initial_velocity = Eigen::Vector3d::Zero());

  void Update(const Eigen::Vector3d& position_measurement,
              const Eigen::Vector3d& control_input,
              double dt);

  Eigen::Vector3d position() const;
  Eigen::Vector3d velocity() const;
  Eigen::Vector3d disturbance() const;
  Eigen::Vector3d error() const;

  void set_omega(double omega_o);
  void set_b0(double b0);

 private:
  std::array<AxisESO, 3> axes_;
};

}  // namespace gpac
}  // namespace quad_rope_lift
