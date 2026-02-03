#pragma once

/// @file wind_disturbance.h
/// @brief Realistic wind disturbance model for outdoor multi-drone simulation.
///
/// Implements the Dryden turbulence model with:
///   - Mean wind vector (constant or time-varying)
///   - Turbulence components (longitudinal, lateral, vertical)
///   - Spatial correlation across multiple drones
///   - Gust model for sudden wind changes

#include <Eigen/Core>
#include <random>
#include <vector>

#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

/// Parameters for the Dryden turbulence model.
struct DrydenTurbulenceParams {
  // Turbulence intensities (standard deviations)
  double sigma_u = 1.0;  ///< Longitudinal turbulence intensity [m/s]
  double sigma_v = 1.0;  ///< Lateral turbulence intensity [m/s]
  double sigma_w = 0.5;  ///< Vertical turbulence intensity [m/s]

  // Turbulence length scales
  double Lu = 200.0;  ///< Longitudinal length scale [m]
  double Lv = 200.0;  ///< Lateral length scale [m]
  double Lw = 50.0;   ///< Vertical length scale [m]

  // Mean wind velocity in world frame
  Eigen::Vector3d mean_wind{2.0, 0.0, 0.0};  ///< Mean wind [m/s]

  // Altitude dependence (low altitude model)
  double altitude_scale_height = 20.0;  ///< Height where full intensity [m]
  bool altitude_dependent = true;       ///< Scale intensity with altitude
};

/// Parameters for wind gust events.
struct GustParams {
  bool enabled = false;           ///< Enable random gusts
  double mean_interval = 30.0;    ///< Mean time between gusts [s]
  double max_magnitude = 5.0;     ///< Maximum gust magnitude [m/s]
  double rise_time = 1.0;         ///< Time to reach peak [s]
  double hold_time = 2.0;         ///< Duration at peak [s]
  double fall_time = 1.5;         ///< Time to decay [s]
};

/// Wind field model for multi-drone simulation.
///
/// Generates realistic wind disturbances affecting multiple drones
/// with proper spatial correlation. Based on Dryden turbulence model
/// (MIL-F-8785C / MIL-HDBK-1797).
///
/// Features:
///   1. Mean wind: Constant or slowly varying background wind
///   2. Turbulence: Stochastic fluctuations with proper spectral content
///   3. Spatial correlation: Drones close together see correlated wind
///   4. Gusts: Optional sudden wind events
///
/// The wind velocity at drone i is:
///   w_i = w_mean + w_turbulence_i + w_gust
///
/// State: Internal turbulence filter states for each drone
///
/// Input ports:
///   - drone_positions: Positions of all drones [3*N_drones]
///
/// Output ports:
///   - wind_velocities: Wind at each drone position [3*N_drones]
///   - wind_forces: Aerodynamic drag forces [3*N_drones] (optional)
///
class WindDisturbance final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WindDisturbance);

  /// Constructs the wind disturbance model.
  ///
  /// @param num_drones Number of drones in the simulation
  /// @param turbulence_params Dryden turbulence parameters
  /// @param gust_params Gust event parameters
  /// @param dt Update timestep [s]
  WindDisturbance(int num_drones,
                  const DrydenTurbulenceParams& turbulence_params = DrydenTurbulenceParams(),
                  const GustParams& gust_params = GustParams(),
                  double dt = 0.01);

  // === Input port accessors ===
  const drake::systems::InputPort<double>& get_drone_positions_input_port() const {
    return get_input_port(positions_port_);
  }

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_wind_velocities_output_port() const {
    return get_output_port(velocities_port_);
  }

  /// Get turbulence parameters.
  const DrydenTurbulenceParams& turbulence_params() const { return turbulence_params_; }

  /// Get gust parameters.
  const GustParams& gust_params() const { return gust_params_; }

  /// Set mean wind velocity.
  void set_mean_wind(const Eigen::Vector3d& wind) {
    turbulence_params_.mean_wind = wind;
  }

  /// Set random seed for reproducibility.
  void set_seed(unsigned int seed);

 private:
  // Discrete update: advance turbulence states
  drake::systems::EventStatus UpdateTurbulence(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Output calculation
  void CalcWindVelocities(const drake::systems::Context<double>& context,
                          drake::systems::BasicVector<double>* output) const;

  // Generate shaped noise for Dryden turbulence
  void GenerateTurbulenceComponent(
      double sigma, double L, double V,
      double dt, double& state, double white_noise) const;

  // Compute gust velocity at current time
  Eigen::Vector3d ComputeGust(double t) const;

  // Compute altitude scaling factor
  double ComputeAltitudeScale(double altitude) const;

  // Compute spatial correlation between two positions
  double ComputeSpatialCorrelation(
      const Eigen::Vector3d& pos1,
      const Eigen::Vector3d& pos2) const;

  // Parameters
  DrydenTurbulenceParams turbulence_params_;
  GustParams gust_params_;
  int num_drones_;
  double dt_;

  // Reference airspeed for Dryden model (affects temporal correlation)
  double reference_airspeed_ = 5.0;  // [m/s]

  // Spatial correlation length scale
  double spatial_correlation_scale_ = 10.0;  // [m]

  // Port indices
  drake::systems::InputPortIndex positions_port_;
  drake::systems::OutputPortIndex velocities_port_;

  // Discrete state index
  // State layout per drone: [u_state, v_state, w_state] = 3 per drone
  // Plus gust state: [gust_active, gust_start_time, gust_direction(3), gust_magnitude]
  drake::systems::DiscreteStateIndex state_index_;

  // State offsets
  static constexpr int kTurbStatePerDrone = 3;
  static constexpr int kGustStateSize = 6;

  // Random number generator (mutable for use in const methods)
  mutable std::mt19937 rng_;
  mutable std::normal_distribution<double> normal_dist_;
  mutable std::exponential_distribution<double> gust_interval_dist_;
};

/// Simple constant wind model for testing.
///
/// Applies a constant wind velocity to all drones.
/// No turbulence, no spatial variation.
///
class ConstantWind final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConstantWind);

  /// Constructs constant wind source.
  ///
  /// @param wind_velocity Constant wind velocity [m/s]
  /// @param num_drones Number of drones
  explicit ConstantWind(const Eigen::Vector3d& wind_velocity,
                        int num_drones = 1);

  const drake::systems::OutputPort<double>& get_wind_output_port() const {
    return get_output_port(wind_port_);
  }

  void set_wind(const Eigen::Vector3d& wind) { wind_velocity_ = wind; }

 private:
  void CalcWind(const drake::systems::Context<double>& context,
                drake::systems::BasicVector<double>* output) const;

  Eigen::Vector3d wind_velocity_;
  int num_drones_;
  drake::systems::OutputPortIndex wind_port_;
};

}  // namespace quad_rope_lift
