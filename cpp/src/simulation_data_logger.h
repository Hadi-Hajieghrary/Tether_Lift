#pragma once

#include <Eigen/Core>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/primitives/vector_log_sink.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tether_lift {

/**
 * @brief Comprehensive data logger for simulation signals.
 *
 * Logs all important simulation data to CSV files in a timestamped folder:
 * - Trajectories (reference and actual for load and drones)
 * - Measurements (GPS, IMU, barometer, tensions, rope states)
 * - Control efforts (forces, torques, desired attitudes)
 * - Estimator outputs and errors (ESO disturbances, concurrent learning params)
 * - GPAC-specific signals (CBF barriers, anti-swing forces, attitude errors)
 * - Wind disturbance (if enabled)
 * - System parameters and configuration
 *
 * All signals are timestamped with simulation time.
 */
class SimulationDataLogger final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimulationDataLogger);

  struct Params {
    std::string base_output_dir;
    double log_period;
    int num_quadcopters;

    // Basic logging flags
    bool log_plant_state;
    bool log_tensions;
    bool log_control_efforts;
    bool log_gps_measurements;
    bool log_estimator_outputs;
    bool log_reference_trajectory;

    // Extended logging flags
    bool log_imu_measurements;
    bool log_barometer_measurements;
    bool log_rope_states;           // Rope stretch, direction, etc.
    bool log_attitude_data;         // Desired/actual attitudes, errors
    bool log_gpac_signals;          // ESO disturbances, CBF barriers, anti-swing
    bool log_wind_disturbance;

    Params()
        : base_output_dir("/workspaces/Tether_Lift/outputs/logs")
        , log_period(0.01)  // 100 Hz
        , num_quadcopters(3)
        , log_plant_state(true)
        , log_tensions(true)
        , log_control_efforts(true)
        , log_gps_measurements(true)
        , log_estimator_outputs(true)
        , log_reference_trajectory(true)
        , log_imu_measurements(true)
        , log_barometer_measurements(true)
        , log_rope_states(true)
        , log_attitude_data(true)
        , log_gpac_signals(true)
        , log_wind_disturbance(true) {}
  };

  /**
   * @brief Construct the data logger.
   *
   * @param plant MultibodyPlant reference for state extraction
   * @param load_body_index Index of the load body
   * @param drone_body_indices Indices of drone bodies
   * @param params Logging parameters
   */
  SimulationDataLogger(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::multibody::BodyIndex load_body_index,
      const std::vector<drake::multibody::BodyIndex>& drone_body_indices,
      const Params& params = Params());

  ~SimulationDataLogger() override;

  // === Basic Input ports ===

  /// Plant state: positions and velocities
  const drake::systems::InputPort<double>& get_plant_state_input() const {
    return get_input_port(plant_state_port_);
  }

  /// Tension input for drone i: [magnitude, fx, fy, fz]
  const drake::systems::InputPort<double>& get_tension_input(int drone_idx) const {
    return get_input_port(tension_ports_.at(drone_idx));
  }

  /// Control effort for drone i: spatial force [tau_x, tau_y, tau_z, fx, fy, fz]
  const drake::systems::InputPort<double>& get_control_effort_input(int drone_idx) const {
    return get_input_port(control_effort_ports_.at(drone_idx));
  }

  /// GPS measurement for drone i: [x, y, z]
  const drake::systems::InputPort<double>& get_gps_input(int drone_idx) const {
    return get_input_port(gps_ports_.at(drone_idx));
  }

  /// Load GPS measurement: [x, y, z]
  const drake::systems::InputPort<double>& get_load_gps_input() const {
    return get_input_port(load_gps_port_);
  }

  /// Estimated state for drone i: [x, y, z, vx, vy, vz]
  const drake::systems::InputPort<double>& get_estimated_state_input(int drone_idx) const {
    return get_input_port(estimated_state_ports_.at(drone_idx));
  }

  /// Load estimated state: [x, y, z, vx, vy, vz]
  const drake::systems::InputPort<double>& get_load_estimated_state_input() const {
    return get_input_port(load_estimated_state_port_);
  }

  /// Reference trajectory: [p_des(3), v_des(3), a_des(3)] = 9D
  const drake::systems::InputPort<double>& get_reference_trajectory_input() const {
    return get_input_port(reference_trajectory_port_);
  }

  // === Extended Input ports ===

  /// IMU measurement for drone i: [ax, ay, az, wx, wy, wz]
  const drake::systems::InputPort<double>& get_imu_input(int drone_idx) const {
    return get_input_port(imu_ports_.at(drone_idx));
  }

  /// Barometer measurement for drone i: [altitude]
  const drake::systems::InputPort<double>& get_barometer_input(int drone_idx) const {
    return get_input_port(barometer_ports_.at(drone_idx));
  }

  /// Rope state for drone i: [stretch, direction_x, direction_y, direction_z, tension_rate]
  const drake::systems::InputPort<double>& get_rope_state_input(int drone_idx) const {
    return get_input_port(rope_state_ports_.at(drone_idx));
  }

  /// Desired attitude for drone i: [qw, qx, qy, qz] or [roll, pitch, yaw]
  const drake::systems::InputPort<double>& get_desired_attitude_input(int drone_idx) const {
    return get_input_port(desired_attitude_ports_.at(drone_idx));
  }

  /// Attitude error for drone i: [err_x, err_y, err_z]
  const drake::systems::InputPort<double>& get_attitude_error_input(int drone_idx) const {
    return get_input_port(attitude_error_ports_.at(drone_idx));
  }

  /// ESO disturbance estimate for drone i: [dx, dy, dz]
  const drake::systems::InputPort<double>& get_eso_disturbance_input(int drone_idx) const {
    return get_input_port(eso_disturbance_ports_.at(drone_idx));
  }

  /// CBF barrier values for drone i: [tautness_lo, tautness_hi, angle, swing, tilt, collision]
  const drake::systems::InputPort<double>& get_cbf_barriers_input(int drone_idx) const {
    return get_input_port(cbf_barrier_ports_.at(drone_idx));
  }

  /// Anti-swing force for drone i: [fx, fy, fz]
  const drake::systems::InputPort<double>& get_antiswing_force_input(int drone_idx) const {
    return get_input_port(antiswing_force_ports_.at(drone_idx));
  }

  /// Concurrent learning parameters for drone i: [theta_1, theta_2, ..., mass_estimate]
  const drake::systems::InputPort<double>& get_concurrent_learning_input(int drone_idx) const {
    return get_input_port(concurrent_learning_ports_.at(drone_idx));
  }

  /// Wind disturbance: [wx, wy, wz] (single global wind)
  const drake::systems::InputPort<double>& get_wind_input() const {
    return get_input_port(wind_port_);
  }

  /// Cable direction for drone i: [qx, qy, qz] (unit vector)
  const drake::systems::InputPort<double>& get_cable_direction_input(int drone_idx) const {
    return get_input_port(cable_direction_ports_.at(drone_idx));
  }

  /**
   * @brief Get the output directory path for this run.
   */
  const std::string& output_dir() const { return output_dir_; }

  /**
   * @brief Write simulation parameters to a config file.
   */
  void WriteConfigFile(const std::map<std::string, std::string>& params);

  /**
   * @brief Finalize logging (flush buffers, close files).
   * Call this after simulation completes.
   */
  void Finalize();

 private:
  drake::systems::EventStatus LogData(
      const drake::systems::Context<double>& context) const;

  void CreateOutputDirectory();
  void OpenLogFiles();
  void WriteHeaders();

  // Helper to get body pose and velocity from plant state
  void ExtractBodyState(
      const drake::systems::Context<double>& plant_context,
      drake::multibody::BodyIndex body_index,
      Eigen::Vector3d* position,
      Eigen::Vector3d* velocity,
      Eigen::Quaterniond* orientation = nullptr,
      Eigen::Vector3d* angular_velocity = nullptr) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Body indices
  drake::multibody::BodyIndex load_body_index_;
  std::vector<drake::multibody::BodyIndex> drone_body_indices_;

  // Parameters
  Params params_;

  // Output directory (timestamped)
  std::string output_dir_;

  // Basic port indices
  int plant_state_port_{-1};
  std::vector<int> tension_ports_;
  std::vector<int> control_effort_ports_;
  std::vector<int> gps_ports_;
  int load_gps_port_{-1};
  std::vector<int> estimated_state_ports_;
  int load_estimated_state_port_{-1};
  int reference_trajectory_port_{-1};

  // Extended port indices
  std::vector<int> imu_ports_;
  std::vector<int> barometer_ports_;
  std::vector<int> rope_state_ports_;
  std::vector<int> desired_attitude_ports_;
  std::vector<int> attitude_error_ports_;
  std::vector<int> eso_disturbance_ports_;
  std::vector<int> cbf_barrier_ports_;
  std::vector<int> antiswing_force_ports_;
  std::vector<int> concurrent_learning_ports_;
  std::vector<int> cable_direction_ports_;
  int wind_port_{-1};

  // Log files (mutable for const logging method)
  mutable std::ofstream trajectory_file_;
  mutable std::ofstream tension_file_;
  mutable std::ofstream control_file_;
  mutable std::ofstream gps_file_;
  mutable std::ofstream estimator_file_;
  mutable std::ofstream reference_file_;
  mutable std::ofstream imu_file_;
  mutable std::ofstream barometer_file_;
  mutable std::ofstream rope_state_file_;
  mutable std::ofstream attitude_file_;
  mutable std::ofstream gpac_file_;
  mutable std::ofstream wind_file_;

  mutable bool files_opened_ = false;
  mutable bool finalized_ = false;
};

}  // namespace tether_lift
