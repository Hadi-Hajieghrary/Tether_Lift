#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for the Error-State Kalman Filter.
struct EskfParams {
  // === Process noise (continuous-time power spectral densities) ===
  /// Accelerometer noise density [m/s²/√Hz]
  Eigen::Vector3d accel_noise_density{4e-3, 4e-3, 4e-3};

  /// Gyroscope noise density [rad/s/√Hz]
  Eigen::Vector3d gyro_noise_density{5e-4, 5e-4, 5e-4};

  /// Accelerometer bias random walk [m/s³/√Hz]
  Eigen::Vector3d accel_bias_random_walk{1e-4, 1e-4, 1e-4};

  /// Gyroscope bias random walk [rad/s²/√Hz]
  Eigen::Vector3d gyro_bias_random_walk{1e-5, 1e-5, 1e-5};

  // === Measurement noise ===
  /// GPS position measurement noise [m]
  Eigen::Vector3d gps_position_noise{0.02, 0.02, 0.05};

  /// GPS velocity measurement noise [m/s] (if available)
  Eigen::Vector3d gps_velocity_noise{0.1, 0.1, 0.1};

  /// Barometer altitude measurement noise [m]
  double baro_altitude_noise = 0.3;

  // === Initial uncertainties ===
  /// Initial position uncertainty [m]
  Eigen::Vector3d initial_position_stddev{0.1, 0.1, 0.1};

  /// Initial velocity uncertainty [m/s]
  Eigen::Vector3d initial_velocity_stddev{0.1, 0.1, 0.1};

  /// Initial attitude uncertainty [rad]
  Eigen::Vector3d initial_attitude_stddev{0.05, 0.05, 0.1};

  /// Initial accelerometer bias uncertainty [m/s²]
  Eigen::Vector3d initial_accel_bias_stddev{0.1, 0.1, 0.1};

  /// Initial gyroscope bias uncertainty [rad/s]
  Eigen::Vector3d initial_gyro_bias_stddev{0.01, 0.01, 0.01};
};

/// 15-state Error-State Kalman Filter for IMU/GPS/Barometer fusion.
///
/// State vector (nominal state, not error state):
///   [position(3), velocity(3), quaternion(4), accel_bias(3), gyro_bias(3)]
///   Total: 16 elements (quaternion has 4 components)
///
/// Error state (for covariance):
///   [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]
///   Total: 15 elements (attitude error is 3-vector)
///
/// The filter propagates using IMU measurements at high rate (prediction),
/// and corrects with GPS/barometer at lower rates (measurement update).
///
/// Input ports:
///   - accel_measurement: Accelerometer reading [m/s²] in body frame (3)
///   - gyro_measurement: Gyroscope reading [rad/s] in body frame (3)
///   - gps_position: GPS position measurement [m] in world frame (3)
///   - gps_valid: GPS measurement validity flag (1)
///   - baro_altitude: Barometer altitude measurement [m] (1)
///   - baro_valid: Barometer measurement validity flag (1)
///
/// Output ports:
///   - estimated_pose: [position(3), quaternion(4)] (7)
///   - estimated_velocity: Linear velocity in world frame (3)
///   - estimated_state: [position(3), velocity(3)] for controller (6)
///   - estimated_biases: [accel_bias(3), gyro_bias(3)] (6)
///   - covariance_diagonal: Diagonal of 15x15 error covariance (15)
///
class EskfEstimator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EskfEstimator);

  /// Constructs the ESKF estimator.
  ///
  /// @param imu_dt IMU propagation time step [s] (typically 0.0025 for 400 Hz)
  /// @param params Filter parameters
  EskfEstimator(double imu_dt, const EskfParams& params = EskfParams());

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_accel_input_port() const {
    return get_input_port(accel_port_);
  }
  const drake::systems::InputPort<double>& get_gyro_input_port() const {
    return get_input_port(gyro_port_);
  }
  const drake::systems::InputPort<double>& get_gps_position_input_port() const {
    return get_input_port(gps_position_port_);
  }
  const drake::systems::InputPort<double>& get_gps_valid_input_port() const {
    return get_input_port(gps_valid_port_);
  }
  const drake::systems::InputPort<double>& get_baro_altitude_input_port() const {
    return get_input_port(baro_altitude_port_);
  }
  const drake::systems::InputPort<double>& get_baro_valid_input_port() const {
    return get_input_port(baro_valid_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_estimated_pose_output_port()
      const {
    return get_output_port(pose_output_port_);
  }
  const drake::systems::OutputPort<double>& get_estimated_velocity_output_port()
      const {
    return get_output_port(velocity_output_port_);
  }
  const drake::systems::OutputPort<double>& get_estimated_state_output_port()
      const {
    return get_output_port(state_output_port_);
  }
  const drake::systems::OutputPort<double>& get_estimated_biases_output_port()
      const {
    return get_output_port(biases_output_port_);
  }
  const drake::systems::OutputPort<double>& get_covariance_output_port() const {
    return get_output_port(covariance_output_port_);
  }

  /// Initialize the filter state.
  ///
  /// @param context Mutable context for this system
  /// @param position Initial position in world frame [m]
  /// @param velocity Initial velocity in world frame [m/s]
  /// @param quaternion Initial orientation as [w, x, y, z]
  /// @param accel_bias Initial accelerometer bias [m/s²]
  /// @param gyro_bias Initial gyroscope bias [rad/s]
  void SetInitialState(drake::systems::Context<double>* context,
                       const Eigen::Vector3d& position,
                       const Eigen::Vector3d& velocity,
                       const Eigen::Vector4d& quaternion,
                       const Eigen::Vector3d& accel_bias = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& gyro_bias = Eigen::Vector3d::Zero()) const;

  /// Get the current estimated quaternion.
  Eigen::Vector4d GetQuaternion(
      const drake::systems::Context<double>& context) const;

  /// Get the current estimated rotation matrix.
  Eigen::Matrix3d GetRotationMatrix(
      const drake::systems::Context<double>& context) const;

  /// Returns the filter parameters.
  const EskfParams& params() const { return params_; }

  // State size constants
  static constexpr int kNominalStateSize = 16;  // pos(3) + vel(3) + quat(4) + ba(3) + bg(3)
  static constexpr int kErrorStateSize = 15;    // pos(3) + vel(3) + att(3) + ba(3) + bg(3)
  static constexpr int kCovarianceSize = kErrorStateSize * kErrorStateSize;

 private:
  // Discrete update: IMU propagation + measurement updates
  drake::systems::EventStatus UpdateEstimate(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // === State extraction helpers ===
  Eigen::Vector3d ExtractPosition(const Eigen::VectorXd& state) const;
  Eigen::Vector3d ExtractVelocity(const Eigen::VectorXd& state) const;
  Eigen::Vector4d ExtractQuaternion(const Eigen::VectorXd& state) const;
  Eigen::Vector3d ExtractAccelBias(const Eigen::VectorXd& state) const;
  Eigen::Vector3d ExtractGyroBias(const Eigen::VectorXd& state) const;
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> ExtractCovariance(
      const Eigen::VectorXd& state) const;

  // === Core ESKF operations ===

  /// IMU propagation (prediction step)
  void Propagate(const Eigen::Vector3d& accel_meas,
                 const Eigen::Vector3d& gyro_meas, double dt,
                 Eigen::VectorXd& nominal_state,
                 Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const;

  /// GPS position measurement update
  void UpdateGpsPosition(
      const Eigen::Vector3d& gps_position, Eigen::VectorXd& nominal_state,
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const;

  /// Barometer altitude measurement update
  void UpdateBaroAltitude(
      double baro_altitude, Eigen::VectorXd& nominal_state,
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const;

  /// Apply error state to nominal state and reset
  void InjectErrorAndReset(
      const Eigen::Matrix<double, kErrorStateSize, 1>& delta_x,
      Eigen::VectorXd& nominal_state,
      Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>& P) const;

  // === Quaternion utilities ===
  Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& q1,
                                     const Eigen::Vector4d& q2) const;
  Eigen::Vector4d QuaternionFromAxisAngle(const Eigen::Vector3d& axis_angle) const;
  Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& q) const;
  Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) const;

  // === Output calculations ===
  void CalcEstimatedPose(const drake::systems::Context<double>& context,
                         drake::systems::BasicVector<double>* output) const;
  void CalcEstimatedVelocity(const drake::systems::Context<double>& context,
                             drake::systems::BasicVector<double>* output) const;
  void CalcEstimatedState(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;
  void CalcEstimatedBiases(const drake::systems::Context<double>& context,
                           drake::systems::BasicVector<double>* output) const;
  void CalcCovariance(const drake::systems::Context<double>& context,
                      drake::systems::BasicVector<double>* output) const;

  // Parameters
  EskfParams params_;
  double imu_dt_;

  // Continuous-time process noise covariance
  Eigen::Matrix<double, 12, 12> Q_continuous_;

  // Port indices
  drake::systems::InputPortIndex accel_port_;
  drake::systems::InputPortIndex gyro_port_;
  drake::systems::InputPortIndex gps_position_port_;
  drake::systems::InputPortIndex gps_valid_port_;
  drake::systems::InputPortIndex baro_altitude_port_;
  drake::systems::InputPortIndex baro_valid_port_;

  drake::systems::OutputPortIndex pose_output_port_;
  drake::systems::OutputPortIndex velocity_output_port_;
  drake::systems::OutputPortIndex state_output_port_;
  drake::systems::OutputPortIndex biases_output_port_;
  drake::systems::OutputPortIndex covariance_output_port_;

  // Discrete state index
  // State: [nominal_state(16), P_flat(225)]
  drake::systems::DiscreteStateIndex state_index_;

  // Gravity
  static constexpr double kGravity = 9.81;
  const Eigen::Vector3d gravity_world_{0.0, 0.0, -kGravity};
};

}  // namespace quad_rope_lift
