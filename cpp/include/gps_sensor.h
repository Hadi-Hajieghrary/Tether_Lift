#pragma once

#include <Eigen/Core>
#include <random>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for GPS sensor noise and behavior.
struct GpsParams {
  Eigen::Vector3d position_noise_stddev{0.02, 0.02, 0.05};  ///< Position noise [m] (x,y,z)
  double sample_period_sec = 0.1;  ///< GPS update rate (default 10 Hz)
  double dropout_probability = 0.0;  ///< Probability of GPS dropout per sample
  unsigned int random_seed = 42;  ///< Seed for reproducibility
};

/// Simulates a GPS sensor that measures body position with noise.
///
/// This system samples the true position of a body at a fixed rate and
/// adds Gaussian noise to simulate real GPS behavior. Optionally supports
/// random dropouts.
///
/// Input ports:
///   - plant_state: Full state vector from MultibodyPlant
///
/// Output ports:
///   - gps_position: Noisy position measurement [x, y, z]
///   - gps_valid: Boolean indicating if measurement is valid (no dropout)
///
class GpsSensor final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GpsSensor);

  /// Constructs a GPS sensor for a specific body.
  ///
  /// @param plant The MultibodyPlant containing the body.
  /// @param body The body whose position is measured.
  /// @param params GPS sensor parameters.
  GpsSensor(const drake::multibody::MultibodyPlant<double>& plant,
            const drake::multibody::RigidBody<double>& body,
            const GpsParams& params = GpsParams());

  /// Returns the plant state input port.
  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  /// Returns the GPS position output port.
  const drake::systems::OutputPort<double>& get_gps_position_output_port() const {
    return get_output_port(position_output_port_);
  }

  /// Returns the GPS valid flag output port.
  const drake::systems::OutputPort<double>& get_gps_valid_output_port() const {
    return get_output_port(valid_output_port_);
  }

  /// Initialize the GPS state with the true body position (called at t=0).
  /// This ensures the first GPS output is valid.
  void InitializeGpsState(drake::systems::Context<double>* context,
                          const Eigen::Vector3d& initial_position) const;

 private:
  // Discrete update: sample true position, add noise, store in state
  drake::systems::EventStatus UpdateGpsMeasurement(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculation: return stored noisy measurement
  void CalcGpsPosition(const drake::systems::Context<double>& context,
                       drake::systems::BasicVector<double>* output) const;

  void CalcGpsValid(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Body to track
  drake::multibody::BodyIndex body_index_;

  // Parameters
  Eigen::Vector3d noise_stddev_;
  double dropout_probability_;

  // Random number generator (mutable for const methods)
  mutable std::default_random_engine generator_;
  mutable std::normal_distribution<double> noise_dist_{0.0, 1.0};
  mutable std::uniform_real_distribution<double> dropout_dist_{0.0, 1.0};

  // Port indices
  int plant_state_port_{-1};
  int position_output_port_{-1};
  int valid_output_port_{-1};

  // Discrete state indices
  int position_state_index_{-1};
  int valid_state_index_{-1};
};

}  // namespace quad_rope_lift
