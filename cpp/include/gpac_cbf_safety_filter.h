#pragma once

/// @file gpac_cbf_safety_filter.h
/// @brief GPAC-enhanced CBF safety filter with tautness constraint.
///
/// Extends the base CBF safety filter with GPAC-specific constraints:
/// 1. Tautness barrier: Ensures cable remains taut (T > T_min)
/// 2. Load stability barrier: Prevents excessive load swing
/// 3. Geometric constraint: Keeps thrust direction aligned
///
/// Key GPAC additions:
/// - Higher-order CBF (HOCBF) for relative degree 2 constraints
/// - Input-to-state safety with ESO disturbance bounds
/// - Exponential CBF for faster constraint satisfaction
/// - Proper ∂T/∂f gradient using cable direction and attitude
/// - Butterworth-filtered tension rate estimation

#include <Eigen/Core>
#include <drake/systems/framework/leaf_system.h>

#include "gpac_math.h"

namespace quad_rope_lift {
namespace gpac {

/// @brief Second-order Butterworth low-pass filter for scalar signals
///
/// Implements a discrete-time Butterworth filter for smoothing noisy
/// derivative estimates (e.g., tension rate from finite differences).
class ButterworthFilter2 {
 public:
  /// @param cutoff_hz Filter cutoff frequency [Hz]
  /// @param sample_rate_hz Sampling rate [Hz]
  explicit ButterworthFilter2(double cutoff_hz = 15.0, double sample_rate_hz = 200.0) {
    Reset(cutoff_hz, sample_rate_hz);
  }

  /// Reset filter state and recompute coefficients
  void Reset(double cutoff_hz, double sample_rate_hz) {
    // Bilinear transform: s = (2/T)*(z-1)/(z+1)
    const double T = 1.0 / sample_rate_hz;
    const double omega_c = 2.0 * M_PI * cutoff_hz;
    const double omega_d = 2.0 / T * std::tan(omega_c * T / 2.0);  // Pre-warped

    // Second-order Butterworth: H(s) = ω_c² / (s² + √2·ω_c·s + ω_c²)
    const double K = omega_d * T / 2.0;
    const double K2 = K * K;
    const double sqrt2_K = std::sqrt(2.0) * K;
    const double denom = 1.0 + sqrt2_K + K2;

    // Coefficients for difference equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
    //                                              - a1*y[n-1] - a2*y[n-2]
    b0_ = K2 / denom;
    b1_ = 2.0 * K2 / denom;
    b2_ = K2 / denom;
    a1_ = 2.0 * (K2 - 1.0) / denom;
    a2_ = (1.0 - sqrt2_K + K2) / denom;

    // Reset state
    x1_ = x2_ = y1_ = y2_ = 0.0;
    initialized_ = false;
  }

  /// Apply filter to new sample
  double Filter(double x) {
    if (!initialized_) {
      // Initialize to avoid transient
      x1_ = x2_ = x;
      y1_ = y2_ = x;
      initialized_ = true;
      return x;
    }

    const double y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;

    // Shift delay line
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = y;

    return y;
  }

  /// Reset filter state without changing coefficients
  void ResetState() {
    x1_ = x2_ = y1_ = y2_ = 0.0;
    initialized_ = false;
  }

 private:
  double b0_{}, b1_{}, b2_{};  // Feedforward coefficients
  double a1_{}, a2_{};         // Feedback coefficients
  double x1_{}, x2_{};         // Input delay line
  double y1_{}, y2_{};         // Output delay line
  bool initialized_{false};
};

/// @brief Parameters for GPAC CBF safety filter
struct GPACCbfParams {
  // Tautness constraint (most critical for cable-suspended loads)
  double min_tension = 2.0;           ///< Minimum cable tension [N]
  double max_tension = 60.0;          ///< Maximum cable tension [N]
  double tautness_alpha = 3.0;        ///< CBF gain for tautness constraint

  // Cable direction constraint
  double max_cable_angle = 0.6;       ///< Max cable angle from vertical [rad]
  double angle_alpha = 2.0;           ///< CBF gain for angle constraint

  // Load swing constraint
  double max_swing_rate = 1.5;        ///< Maximum cable angular velocity [rad/s]
  double swing_alpha = 1.5;           ///< CBF gain for swing constraint

  // Thrust direction constraint
  double max_tilt = 0.5;              ///< Maximum quadcopter tilt [rad]
  double tilt_alpha = 4.0;            ///< CBF gain for tilt constraint

  // Collision avoidance
  double min_separation = 0.8;        ///< Minimum inter-drone distance [m]
  double collision_alpha = 3.0;       ///< CBF gain for collision avoidance

  // QP parameters
  double qp_relaxation_weight = 100.0; ///< Slack variable weight
  double nominal_tracking_weight = 1.0; ///< Nominal control tracking weight

  // Actuator limits
  double min_thrust = 0.0;
  double max_thrust = 100.0;
  double max_torque = 10.0;

  // ESO-based robustness margin
  double disturbance_margin = 2.0;    ///< Base safety margin [m/s²]
  double eso_disturbance_coupling = 1.5;  ///< Multiplier for ESO disturbance bound

  // Tension rate estimation
  bool use_filtered_tension_rate = true;  ///< Enable Butterworth-filtered Ṫ
  double tension_filter_cutoff_hz = 15.0; ///< Butterworth cutoff frequency [Hz]
  double tension_filter_sample_hz = 200.0; ///< Expected sample rate [Hz]

  // Cable/rope physical parameters (for proper gradient computation)
  double cable_damping = 15.0;        ///< Cable damping coefficient [N·s/m]
};

/// @brief Barrier function values for logging
struct BarrierValues {
  double tautness_lower;  ///< h_T = T - T_min
  double tautness_upper;  ///< h_T = T_max - T
  double cable_angle;     ///< h_θ = cos(θ_max) - cos(θ)
  double swing_rate;      ///< h_ω = ω_max² - ||ω_q||²
  double tilt;            ///< h_tilt = cos(tilt_max) - cos(tilt)
  double collision;       ///< h_col = ||Δr||² - d_min²
};

/// @brief GPAC CBF Safety Filter
///
/// Implements safety filter using Control Barrier Functions with
/// GPAC-specific constraints for cable-suspended load transport.
///
/// The QP solved at each timestep:
///   min  ||f - f_nom||² + λ||δ||²
///   s.t. ḣ_i(x, f) + α_i h_i(x) ≥ -δ_i  for all barriers
///        f_min ≤ f ≤ f_max
///
/// Key improvements over basic CBF:
/// - Correct gradient ∂T/∂f = (c/m)·(q^T R e_3) using cable direction & attitude
/// - Butterworth-filtered tension rate estimation for smooth ḣ
/// - ESO disturbance coupling for robust margins
///
/// Input ports:
///   - nominal_force (3D): Nominal thrust vector [N]
///   - drone_state (13D): [p(3), q(4), v(3), ω(3)]
///   - cable_tension (1D): Current tension [N]
///   - cable_direction (3D): Unit vector load→drone
///   - cable_velocity (3D): Cable angular velocity (optional)
///   - load_position (3D): Load position (optional)
///   - disturbance_bound (1D): ESO disturbance magnitude bound (optional)
///
/// Output ports:
///   - filtered_force (3D): Safe thrust vector [N]
///   - barrier_values (6D): Current barrier function values
///   - constraint_margins (6D): Distance to constraint boundary
///   - tension_rate (1D): Estimated tension rate [N/s]
class GPACCbfSafetyFilter final
    : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GPACCbfSafetyFilter);

  explicit GPACCbfSafetyFilter(
      double drone_mass = 1.5,
      const GPACCbfParams& params = GPACCbfParams());

  // === Input port accessors ===

  const drake::systems::InputPort<double>& get_nominal_force_input() const {
    return get_input_port(nominal_force_port_);
  }

  const drake::systems::InputPort<double>& get_drone_state_input() const {
    return get_input_port(drone_state_port_);
  }

  const drake::systems::InputPort<double>& get_cable_tension_input() const {
    return get_input_port(cable_tension_port_);
  }

  const drake::systems::InputPort<double>& get_cable_direction_input() const {
    return get_input_port(cable_direction_port_);
  }

  const drake::systems::InputPort<double>& get_cable_velocity_input() const {
    return get_input_port(cable_velocity_port_);
  }

  const drake::systems::InputPort<double>& get_disturbance_bound_input() const {
    return get_input_port(disturbance_bound_port_);
  }

  // === Output port accessors ===

  const drake::systems::OutputPort<double>& get_filtered_force_output() const {
    return get_output_port(filtered_force_port_);
  }

  const drake::systems::OutputPort<double>& get_barrier_values_output() const {
    return get_output_port(barrier_values_port_);
  }

  const drake::systems::OutputPort<double>& get_constraint_margins_output() const {
    return get_output_port(constraint_margins_port_);
  }

  const drake::systems::OutputPort<double>& get_tension_rate_output() const {
    return get_output_port(tension_rate_port_);
  }

  // === Parameter access ===

  GPACCbfParams& mutable_params() { return params_; }
  const GPACCbfParams& params() const { return params_; }
  void set_mass(double mass) { drone_mass_ = mass; }

 private:
  void CalcFilteredForce(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcBarrierValues(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcConstraintMargins(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  void CalcTensionRate(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* output) const;

  /// Compute all barrier function values
  BarrierValues ComputeBarriers(
      const drake::systems::Context<double>& context) const;

  /// Compute filtered tension rate estimate
  double ComputeTensionRate(
      const drake::systems::Context<double>& context) const;

  /// Compute proper tension gradient ∂T/∂f
  /// Uses cable direction and drone attitude for correct Lie derivative
  Eigen::Vector3d ComputeTensionGradient(
      const Eigen::Vector3d& cable_dir,
      const Eigen::Matrix3d& R) const;

  /// Get effective disturbance margin including ESO coupling
  double GetEffectiveDisturbanceMargin(
      const drake::systems::Context<double>& context) const;

  /// Solve QP with CBF constraints
  /// Returns filtered force vector
  Eigen::Vector3d SolveQP(
      const Eigen::Vector3d& f_nom,
      const BarrierValues& h,
      double T_dot,
      const Eigen::Vector3d& position,
      const Eigen::Vector3d& velocity,
      double tension,
      const Eigen::Vector3d& cable_dir,
      const Eigen::Matrix3d& R,
      double dist_margin) const;

  /// Simple gradient projection for single constraint
  Eigen::Vector3d ProjectToSafe(
      const Eigen::Vector3d& f_nom,
      double h,
      double h_dot,
      const Eigen::Vector3d& grad_h,
      double alpha,
      double margin) const;

  GPACCbfParams params_;
  double drone_mass_;

  // Tension rate estimation state (mutable for const methods)
  mutable ButterworthFilter2 tension_filter_;
  mutable double prev_tension_{0.0};
  mutable double prev_time_{-1.0};

  // Port indices
  int nominal_force_port_{};
  int drone_state_port_{};
  int cable_tension_port_{};
  int cable_direction_port_{};
  int cable_velocity_port_{};
  int disturbance_bound_port_{};
  int filtered_force_port_{};
  int barrier_values_port_{};
  int constraint_margins_port_{};
  int tension_rate_port_{};
};

}  // namespace gpac
}  // namespace quad_rope_lift
