#pragma once

#include <Eigen/Core>
#include <random>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for barometric altitude sensor.
/// Models three-component noise: bias + correlated (Gauss-Markov) + white.
struct BarometerParams {
  /// White noise standard deviation [m]
  double white_noise_stddev = 0.3;

  /// Correlated noise standard deviation [m] (Gauss-Markov)
  double correlated_noise_stddev = 0.2;

  /// Correlation time constant [s] for Gauss-Markov noise
  double correlation_time = 5.0;

  /// Slow bias drift rate [m/s] (e.g., due to weather changes)
  double bias_drift_rate = 0.002;  // ~0.1 m/min

  /// Initial bias [m]
  double initial_bias = 0.0;

  /// Sample period [s] (default 25 Hz)
  double sample_period_sec = 0.04;

  /// Quantization resolution [m] (typical MEMS: 0.1-0.3 m)
  double resolution = 0.1;

  /// Random seed for reproducibility
  unsigned int random_seed = 42;
};

/// Simulates a barometric altitude sensor.
///
/// Models realistic noise characteristics including:
/// - White noise (measurement noise)
/// - Correlated noise (first-order Gauss-Markov process)
/// - Slow bias drift (weather-related)
/// - Quantization effects
///
/// The barometer measures altitude (z-position) with respect to a reference.
///
/// Input ports:
///   - plant_state: Full state vector from MultibodyPlant
///
/// Output ports:
///   - altitude: Noisy altitude measurement [m] (1)
///   - baro_valid: Boolean indicating measurement validity (1)
///
class BarometerSensor final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BarometerSensor);

  /// Constructs a barometer sensor attached to a specific body.
  ///
  /// @param plant The MultibodyPlant containing the body.
  /// @param body The body the barometer is attached to.
  /// @param params Barometer noise parameters.
  BarometerSensor(const drake::multibody::MultibodyPlant<double>& plant,
                  const drake::multibody::RigidBody<double>& body,
                  const BarometerParams& params = BarometerParams());

  /// Returns the plant state input port.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Returns the altitude measurement output port.
  const drake::systems::OutputPort<double>& get_altitude_output_port() const {
    return get_output_port(altitude_output_port_);
  }

  /// Returns the barometer valid flag output port.
  const drake::systems::OutputPort<double>& get_baro_valid_output_port() const {
    return get_output_port(valid_output_port_);
  }

  /// Initialize barometer state (called after diagram is built).
  void InitializeState(drake::systems::Context<double>* context,
                       double initial_altitude) const;

  /// Get current bias value (for debugging/logging).
  double GetBias(const drake::systems::Context<double>& context) const;

  /// Get current correlated noise state (for debugging/logging).
  double GetCorrelatedNoise(
      const drake::systems::Context<double>& context) const;

  /// Returns the barometer parameters.
  const BarometerParams& params() const { return params_; }

 private:
  // Discrete update: sample true altitude, add noise components
  drake::systems::EventStatus UpdateBaroMeasurement(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculations
  void CalcAltitude(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;

  void CalcBaroValid(const drake::systems::Context<double>& context,
                     drake::systems::BasicVector<double>* output) const;

  // Apply quantization
  double Quantize(double value) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::multibody::BodyIndex body_index_;

  // Parameters
  BarometerParams params_;

  // Port indices
  drake::systems::InputPortIndex plant_state_port_;
  drake::systems::OutputPortIndex altitude_output_port_;
  drake::systems::OutputPortIndex valid_output_port_;

  // Discrete state indices
  // State layout: [altitude_meas(1), bias(1), correlated_noise(1), valid(1)]
  drake::systems::DiscreteStateIndex measurement_state_index_;

  // Random number generators (mutable for const methods)
  mutable std::mt19937 generator_;
  mutable std::normal_distribution<double> noise_dist_{0.0, 1.0};
};

/// Factory function to create high-quality barometer parameters (e.g., DPS310).
BarometerParams HighQualityBarometerParams();

/// Factory function to create standard barometer parameters (e.g., BMP280).
BarometerParams StandardBarometerParams();

}  // namespace quad_rope_lift
