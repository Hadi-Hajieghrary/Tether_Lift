#include "simulation_data_logger.h"

#include <drake/math/rigid_transform.h>
#include <drake/multibody/tree/rigid_body.h>

#include <ctime>
#include <iostream>

namespace tether_lift {

using drake::math::RigidTransformd;
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::SpatialVelocity;
using drake::systems::Context;
using drake::systems::EventStatus;

SimulationDataLogger::SimulationDataLogger(
    const MultibodyPlant<double>& plant,
    BodyIndex load_body_index,
    const std::vector<BodyIndex>& drone_body_indices,
    const Params& params)
    : plant_(plant),
      load_body_index_(load_body_index),
      drone_body_indices_(drone_body_indices),
      params_(params) {

  const int num_drones = static_cast<int>(drone_body_indices_.size());

  // Create output directory with timestamp
  CreateOutputDirectory();

  // Declare input ports

  // Plant state
  plant_state_port_ = this->DeclareVectorInputPort(
      "plant_state", plant_.num_positions() + plant_.num_velocities())
      .get_index();

  // Tension inputs (one per drone): [magnitude, fx, fy, fz]
  tension_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    tension_ports_[i] = this->DeclareVectorInputPort(
        "tension_" + std::to_string(i), 4).get_index();
  }

  // Control effort inputs (one per drone): spatial force 6D
  control_effort_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    control_effort_ports_[i] = this->DeclareVectorInputPort(
        "control_effort_" + std::to_string(i), 6).get_index();
  }

  // GPS inputs (one per drone + load): [x, y, z]
  gps_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    gps_ports_[i] = this->DeclareVectorInputPort(
        "gps_" + std::to_string(i), 3).get_index();
  }
  load_gps_port_ = this->DeclareVectorInputPort("load_gps", 3).get_index();

  // Estimated state inputs (one per drone + load): [x, y, z, vx, vy, vz]
  estimated_state_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    estimated_state_ports_[i] = this->DeclareVectorInputPort(
        "estimated_state_" + std::to_string(i), 6).get_index();
  }
  load_estimated_state_port_ = this->DeclareVectorInputPort(
      "load_estimated_state", 6).get_index();

  // Reference trajectory: [p_des(3), v_des(3), a_des(3)]
  reference_trajectory_port_ = this->DeclareVectorInputPort(
      "reference_trajectory", 9).get_index();

  // Declare periodic publish event for logging
  this->DeclarePeriodicPublishEvent(
      params_.log_period,
      0.0,
      &SimulationDataLogger::LogData);

  std::cout << "Data logger initialized. Output directory: " << output_dir_ << std::endl;
}

SimulationDataLogger::~SimulationDataLogger() {
  Finalize();
}

void SimulationDataLogger::CreateOutputDirectory() {
  // Get current timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm* tm_now = std::localtime(&time_t_now);

  std::ostringstream timestamp;
  timestamp << std::put_time(tm_now, "%Y%m%d_%H%M%S");

  output_dir_ = params_.base_output_dir + "/" + timestamp.str();

  // Create directory
  std::filesystem::create_directories(output_dir_);
}

void SimulationDataLogger::OpenLogFiles() {
  if (files_opened_) return;

  if (params_.log_plant_state) {
    trajectory_file_.open(output_dir_ + "/trajectories.csv");
  }
  if (params_.log_tensions) {
    tension_file_.open(output_dir_ + "/tensions.csv");
  }
  if (params_.log_control_efforts) {
    control_file_.open(output_dir_ + "/control_efforts.csv");
  }
  if (params_.log_gps_measurements) {
    gps_file_.open(output_dir_ + "/gps_measurements.csv");
  }
  if (params_.log_estimator_outputs) {
    estimator_file_.open(output_dir_ + "/estimator_outputs.csv");
  }
  if (params_.log_reference_trajectory) {
    reference_file_.open(output_dir_ + "/reference_trajectory.csv");
  }

  WriteHeaders();
  files_opened_ = true;
}

void SimulationDataLogger::WriteHeaders() {
  const int num_drones = static_cast<int>(drone_body_indices_.size());

  // Trajectory file header
  if (trajectory_file_.is_open()) {
    trajectory_file_ << "time";
    trajectory_file_ << ",load_x,load_y,load_z,load_vx,load_vy,load_vz";
    for (int i = 0; i < num_drones; ++i) {
      trajectory_file_ << ",drone" << i << "_x,drone" << i << "_y,drone" << i << "_z";
      trajectory_file_ << ",drone" << i << "_vx,drone" << i << "_vy,drone" << i << "_vz";
    }
    trajectory_file_ << "\n";
  }

  // Tension file header
  if (tension_file_.is_open()) {
    tension_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      tension_file_ << ",rope" << i << "_mag,rope" << i << "_fx,rope" << i << "_fy,rope" << i << "_fz";
    }
    tension_file_ << "\n";
  }

  // Control file header
  if (control_file_.is_open()) {
    control_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      control_file_ << ",drone" << i << "_tau_x,drone" << i << "_tau_y,drone" << i << "_tau_z";
      control_file_ << ",drone" << i << "_f_x,drone" << i << "_f_y,drone" << i << "_f_z";
    }
    control_file_ << "\n";
  }

  // GPS file header
  if (gps_file_.is_open()) {
    gps_file_ << "time";
    gps_file_ << ",load_gps_x,load_gps_y,load_gps_z";
    for (int i = 0; i < num_drones; ++i) {
      gps_file_ << ",drone" << i << "_gps_x,drone" << i << "_gps_y,drone" << i << "_gps_z";
    }
    gps_file_ << "\n";
  }

  // Estimator file header
  if (estimator_file_.is_open()) {
    estimator_file_ << "time";
    estimator_file_ << ",load_est_x,load_est_y,load_est_z,load_est_vx,load_est_vy,load_est_vz";
    for (int i = 0; i < num_drones; ++i) {
      estimator_file_ << ",drone" << i << "_est_x,drone" << i << "_est_y,drone" << i << "_est_z";
      estimator_file_ << ",drone" << i << "_est_vx,drone" << i << "_est_vy,drone" << i << "_est_vz";
    }
    estimator_file_ << "\n";
  }

  // Reference trajectory file header
  if (reference_file_.is_open()) {
    reference_file_ << "time";
    reference_file_ << ",ref_x,ref_y,ref_z,ref_vx,ref_vy,ref_vz,ref_ax,ref_ay,ref_az";
    reference_file_ << "\n";
  }
}

void SimulationDataLogger::ExtractBodyState(
    const Context<double>& plant_context,
    BodyIndex body_index,
    Eigen::Vector3d* position,
    Eigen::Vector3d* velocity) const {
  const auto& body = plant_.get_body(body_index);
  RigidTransformd X_WB = plant_.EvalBodyPoseInWorld(plant_context, body);
  *position = X_WB.translation();

  SpatialVelocity<double> V_WB = plant_.EvalBodySpatialVelocityInWorld(
      plant_context, body);
  *velocity = V_WB.translational();
}

EventStatus SimulationDataLogger::LogData(
    const Context<double>& context) const {

  if (finalized_) return EventStatus::Succeeded();

  // Open files on first call (lazy initialization)
  if (!files_opened_) {
    const_cast<SimulationDataLogger*>(this)->OpenLogFiles();
  }

  const double time = context.get_time();
  const int num_drones = static_cast<int>(drone_body_indices_.size());

  // Get plant state and create plant context
  const auto& state_vector = get_plant_state_input().Eval(context);
  auto plant_context = plant_.CreateDefaultContext();
  plant_.SetPositionsAndVelocities(plant_context.get(), state_vector);

  // === Log trajectories (ground truth) ===
  if (trajectory_file_.is_open()) {
    trajectory_file_ << std::fixed << std::setprecision(6) << time;

    // Load state
    Eigen::Vector3d load_pos, load_vel;
    ExtractBodyState(*plant_context, load_body_index_, &load_pos, &load_vel);
    trajectory_file_ << "," << load_pos.x() << "," << load_pos.y() << "," << load_pos.z();
    trajectory_file_ << "," << load_vel.x() << "," << load_vel.y() << "," << load_vel.z();

    // Drone states
    for (int i = 0; i < num_drones; ++i) {
      Eigen::Vector3d drone_pos, drone_vel;
      ExtractBodyState(*plant_context, drone_body_indices_[i], &drone_pos, &drone_vel);
      trajectory_file_ << "," << drone_pos.x() << "," << drone_pos.y() << "," << drone_pos.z();
      trajectory_file_ << "," << drone_vel.x() << "," << drone_vel.y() << "," << drone_vel.z();
    }
    trajectory_file_ << "\n";
  }

  // === Log tensions ===
  if (tension_file_.is_open()) {
    tension_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(tension_ports_[i]).HasValue(context)) {
        const auto& tension = get_tension_input(i).Eval(context);
        tension_file_ << "," << tension(0) << "," << tension(1)
                      << "," << tension(2) << "," << tension(3);
      } else {
        tension_file_ << ",0,0,0,0";
      }
    }
    tension_file_ << "\n";
  }

  // === Log control efforts ===
  if (control_file_.is_open()) {
    control_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(control_effort_ports_[i]).HasValue(context)) {
        const auto& effort = get_control_effort_input(i).Eval(context);
        control_file_ << "," << effort(0) << "," << effort(1) << "," << effort(2);
        control_file_ << "," << effort(3) << "," << effort(4) << "," << effort(5);
      } else {
        control_file_ << ",0,0,0,0,0,0";
      }
    }
    control_file_ << "\n";
  }

  // === Log GPS measurements ===
  if (gps_file_.is_open()) {
    gps_file_ << std::fixed << std::setprecision(6) << time;

    // Load GPS
    if (get_input_port(load_gps_port_).HasValue(context)) {
      const auto& load_gps = get_load_gps_input().Eval(context);
      gps_file_ << "," << load_gps(0) << "," << load_gps(1) << "," << load_gps(2);
    } else {
      gps_file_ << ",0,0,0";
    }

    // Drone GPS
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(gps_ports_[i]).HasValue(context)) {
        const auto& gps = get_gps_input(i).Eval(context);
        gps_file_ << "," << gps(0) << "," << gps(1) << "," << gps(2);
      } else {
        gps_file_ << ",0,0,0";
      }
    }
    gps_file_ << "\n";
  }

  // === Log estimator outputs ===
  if (estimator_file_.is_open()) {
    estimator_file_ << std::fixed << std::setprecision(6) << time;

    // Load estimate
    if (get_input_port(load_estimated_state_port_).HasValue(context)) {
      const auto& load_est = get_load_estimated_state_input().Eval(context);
      estimator_file_ << "," << load_est(0) << "," << load_est(1) << "," << load_est(2);
      estimator_file_ << "," << load_est(3) << "," << load_est(4) << "," << load_est(5);
    } else {
      estimator_file_ << ",0,0,0,0,0,0";
    }

    // Drone estimates
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(estimated_state_ports_[i]).HasValue(context)) {
        const auto& est = get_estimated_state_input(i).Eval(context);
        estimator_file_ << "," << est(0) << "," << est(1) << "," << est(2);
        estimator_file_ << "," << est(3) << "," << est(4) << "," << est(5);
      } else {
        estimator_file_ << ",0,0,0,0,0,0";
      }
    }
    estimator_file_ << "\n";
  }

  // === Log reference trajectory ===
  if (reference_file_.is_open()) {
    reference_file_ << std::fixed << std::setprecision(6) << time;

    if (get_input_port(reference_trajectory_port_).HasValue(context)) {
      const auto& ref = get_reference_trajectory_input().Eval(context);
      for (int i = 0; i < 9; ++i) {
        reference_file_ << "," << ref(i);
      }
    } else {
      reference_file_ << ",0,0,0,0,0,0,0,0,0";
    }
    reference_file_ << "\n";
  }

  return EventStatus::Succeeded();
}

void SimulationDataLogger::WriteConfigFile(
    const std::map<std::string, std::string>& params) {
  std::ofstream config_file(output_dir_ + "/config.txt");
  if (!config_file.is_open()) {
    std::cerr << "Warning: Could not open config file for writing." << std::endl;
    return;
  }

  config_file << "# Simulation Configuration\n";
  config_file << "# Generated at: " << output_dir_.substr(output_dir_.rfind('/') + 1) << "\n\n";

  for (const auto& [key, value] : params) {
    config_file << key << " = " << value << "\n";
  }

  config_file.close();
  std::cout << "Configuration saved to: " << output_dir_ << "/config.txt" << std::endl;
}

void SimulationDataLogger::Finalize() {
  if (finalized_) return;

  if (trajectory_file_.is_open()) trajectory_file_.close();
  if (tension_file_.is_open()) tension_file_.close();
  if (control_file_.is_open()) control_file_.close();
  if (gps_file_.is_open()) gps_file_.close();
  if (estimator_file_.is_open()) estimator_file_.close();
  if (reference_file_.is_open()) reference_file_.close();

  finalized_ = true;
  std::cout << "Data logging finalized. Files saved to: " << output_dir_ << std::endl;
}

}  // namespace tether_lift
