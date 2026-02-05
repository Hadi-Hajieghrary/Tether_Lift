#pragma once

#include <Eigen/Core>
#include <random>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for IMU sensor noise characteristics.
/// Based on typical MEMS IMU specifications (consumer to tactical grade).
struct ImuParams {
  // === Gyroscope parameters ===
  /// Angular random walk (ARW) [rad/s/√Hz] - white noise density
  /// Consumer: 0.0005-0.001, Tactical: 0.00003
  Eigen::Vector3d gyro_noise_density{5e-4, 5e-4, 5e-4};

  /// Gyroscope bias instability [rad/s]
  /// Consumer: 1e-4 (≈20°/hr), Tactical: 1e-6
  Eigen::Vector3d gyro_bias_instability{1e-4, 1e-4, 1e-4};

  /// Gyroscope bias correlation time [s] (first-order Gauss-Markov)
  double gyro_bias_time_constant = 3600.0;

  /// Initial gyroscope bias [rad/s]
  Eigen::Vector3d gyro_initial_bias{0.0, 0.0, 0.0};

  // === Accelerometer parameters ===
  /// Velocity random walk (VRW) [m/s²/√Hz] - white noise density
  /// Consumer: 0.003-0.005 (300-500 μg/√Hz), Tactical: 0.0002
  Eigen::Vector3d accel_noise_density{4e-3, 4e-3, 4e-3};

  /// Accelerometer bias instability [m/s²]
  /// Consumer: 1e-3 (≈100 μg), Tactical: 1e-5
  Eigen::Vector3d accel_bias_instability{1e-3, 1e-3, 1e-3};

  /// Accelerometer bias correlation time [s]
  double accel_bias_time_constant = 3600.0;

  /// Initial accelerometer bias [m/s²]
  Eigen::Vector3d accel_initial_bias{0.0, 0.0, 0.0};

  // === Timing ===
  /// IMU sample period [s] (default 400 Hz)
  double sample_period_sec = 0.0025;

  /// Random seed for reproducibility
  unsigned int random_seed = 42;
};

/// Simulates a 6-DOF IMU (3-axis gyroscope + 3-axis accelerometer).
///
/// Models realistic noise characteristics including:
/// - White noise (angular/velocity random walk)
/// - Bias with first-order Gauss-Markov dynamics
/// - Proper coordinate frame transformations
///
/// The accelerometer measures specific force (acceleration - gravity) in body frame.
/// The gyroscope measures angular velocity in body frame.
///
/// Input ports:
///   - plant_state: Full state vector from MultibodyPlant
///
/// Output ports:
///   - gyro_measurement: Angular velocity measurement [rad/s] in body frame (3)
///   - accel_measurement: Specific force measurement [m/s²] in body frame (3)
///   - imu_valid: Boolean indicating measurement validity (1)
///
class ImuSensor final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ImuSensor);

  /// Constructs an IMU sensor attached to a specific body.
  ///
  /// @param plant The MultibodyPlant containing the body.
  /// @param body The body the IMU is attached to.
  /// @param params IMU noise parameters.
  ImuSensor(const drake::multibody::MultibodyPlant<double>& plant,
            const drake::multibody::RigidBody<double>& body,
            const ImuParams& params = ImuParams());

  /// Returns the plant state input port.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Returns the gyroscope measurement output port.
  const drake::systems::OutputPort<double>& get_gyro_output_port() const {
    return get_output_port(gyro_output_port_);
  }

  /// Returns the accelerometer measurement output port.
  const drake::systems::OutputPort<double>& get_accel_output_port() const {
    return get_output_port(accel_output_port_);
  }

  /// Returns the IMU valid flag output port.
  const drake::systems::OutputPort<double>& get_imu_valid_output_port() const {
    return get_output_port(valid_output_port_);
  }

  /// Initialize IMU biases (called after diagram is built).
  void InitializeBiases(drake::systems::Context<double>* context,
                        const Eigen::Vector3d& gyro_bias,
                        const Eigen::Vector3d& accel_bias) const;

  /// Get current bias estimates (for debugging/logging).
  Eigen::Vector3d GetGyroBias(
      const drake::systems::Context<double>& context) const;
  Eigen::Vector3d GetAccelBias(
      const drake::systems::Context<double>& context) const;

  /// Returns the IMU parameters.
  const ImuParams& params() const { return params_; }

 private:
  // Discrete update: sample true state, add noise and bias, store measurements
  drake::systems::EventStatus UpdateImuMeasurement(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculations
  void CalcGyroMeasurement(const drake::systems::Context<double>& context,
                           drake::systems::BasicVector<double>* output) const;

  void CalcAccelMeasurement(const drake::systems::Context<double>& context,
                            drake::systems::BasicVector<double>* output) const;

  void CalcImuValid(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::multibody::BodyIndex body_index_;

  // Parameters
  ImuParams params_;

  // Port indices
  drake::systems::InputPortIndex plant_state_port_;
  drake::systems::OutputPortIndex gyro_output_port_;
  drake::systems::OutputPortIndex accel_output_port_;
  drake::systems::OutputPortIndex valid_output_port_;

  // Discrete state indices
  // State layout: [gyro_meas(3), accel_meas(3), gyro_bias(3), accel_bias(3), valid(1)]
  drake::systems::DiscreteStateIndex measurement_state_index_;

  // Random number generators (mutable for const methods)
  mutable std::mt19937 generator_;
  mutable std::normal_distribution<double> noise_dist_{0.0, 1.0};

  // Gravity vector in world frame
  static constexpr double kGravity = 9.81;
  const Eigen::Vector3d gravity_world_{0.0, 0.0, -kGravity};
};

/// Factory function to create consumer-grade IMU parameters (e.g., MPU6050).
ImuParams ConsumerImuParams();

/// Factory function to create tactical-grade IMU parameters (e.g., ADIS16488).
ImuParams TacticalImuParams();

}  // namespace quad_rope_lift
