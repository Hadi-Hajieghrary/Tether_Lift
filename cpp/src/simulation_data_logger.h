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
 * - Measurements (GPS, tensions, rope states)
 * - Control efforts (forces, torques)
 * - Estimator outputs and errors
 * - System parameters and configuration
 */
class SimulationDataLogger final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimulationDataLogger);

  struct Params {
    std::string base_output_dir;
    double log_period;
    int num_quadcopters;
    bool log_plant_state;
    bool log_tensions;
    bool log_control_efforts;
    bool log_gps_measurements;
    bool log_estimator_outputs;
    bool log_reference_trajectory;

    Params()
        : base_output_dir("/workspaces/Tether_Lift/outputs/logs")
        , log_period(0.01)  // 100 Hz
        , num_quadcopters(3)
        , log_plant_state(true)
        , log_tensions(true)
        , log_control_efforts(true)
        , log_gps_measurements(true)
        , log_estimator_outputs(true)
        , log_reference_trajectory(true) {}
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

  // === Input ports ===

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
      Eigen::Vector3d* velocity) const;

  // Plant reference
  const drake::multibody::MultibodyPlant<double>& plant_;

  // Body indices
  drake::multibody::BodyIndex load_body_index_;
  std::vector<drake::multibody::BodyIndex> drone_body_indices_;

  // Parameters
  Params params_;

  // Output directory (timestamped)
  std::string output_dir_;

  // Port indices
  int plant_state_port_{-1};
  std::vector<int> tension_ports_;
  std::vector<int> control_effort_ports_;
  std::vector<int> gps_ports_;
  int load_gps_port_{-1};
  std::vector<int> estimated_state_ports_;
  int load_estimated_state_port_{-1};
  int reference_trajectory_port_{-1};

  // Log files (mutable for const logging method)
  mutable std::ofstream trajectory_file_;
  mutable std::ofstream tension_file_;
  mutable std::ofstream control_file_;
  mutable std::ofstream gps_file_;
  mutable std::ofstream estimator_file_;
  mutable std::ofstream reference_file_;

  mutable bool files_opened_ = false;
  mutable bool finalized_ = false;
};

}  // namespace tether_lift
