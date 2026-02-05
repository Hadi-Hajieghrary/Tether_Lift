#include "simulation_data_logger.h"

#include <drake/math/rigid_transform.h>
#include <drake/math/roll_pitch_yaw.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/tree/rigid_body.h>

#include <ctime>
#include <iostream>

namespace tether_lift {

using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
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

  // =========================================================================
  // Declare Basic Input Ports
  // =========================================================================

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

  // =========================================================================
  // Declare Extended Input Ports
  // =========================================================================

  // IMU inputs (one per drone): [ax, ay, az, wx, wy, wz]
  imu_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    imu_ports_[i] = this->DeclareVectorInputPort(
        "imu_" + std::to_string(i), 6).get_index();
  }

  // Barometer inputs (one per drone): [altitude]
  barometer_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    barometer_ports_[i] = this->DeclareVectorInputPort(
        "barometer_" + std::to_string(i), 1).get_index();
  }

  // Rope state inputs: [stretch, dir_x, dir_y, dir_z, tension_rate]
  rope_state_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    rope_state_ports_[i] = this->DeclareVectorInputPort(
        "rope_state_" + std::to_string(i), 5).get_index();
  }

  // Desired attitude inputs: [qw, qx, qy, qz]
  desired_attitude_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    desired_attitude_ports_[i] = this->DeclareVectorInputPort(
        "desired_attitude_" + std::to_string(i), 4).get_index();
  }

  // Attitude error inputs: [err_x, err_y, err_z]
  attitude_error_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    attitude_error_ports_[i] = this->DeclareVectorInputPort(
        "attitude_error_" + std::to_string(i), 3).get_index();
  }

  // ESO disturbance estimates: [dx, dy, dz]
  eso_disturbance_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    eso_disturbance_ports_[i] = this->DeclareVectorInputPort(
        "eso_disturbance_" + std::to_string(i), 3).get_index();
  }

  // CBF barrier values: [tautness_lo, tautness_hi, angle, swing, tilt, collision]
  cbf_barrier_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    cbf_barrier_ports_[i] = this->DeclareVectorInputPort(
        "cbf_barriers_" + std::to_string(i), 6).get_index();
  }

  // Anti-swing force: [fx, fy, fz]
  antiswing_force_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    antiswing_force_ports_[i] = this->DeclareVectorInputPort(
        "antiswing_force_" + std::to_string(i), 3).get_index();
  }

  // Concurrent learning parameters: [theta..., mass_estimate] - 4D for now
  concurrent_learning_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    concurrent_learning_ports_[i] = this->DeclareVectorInputPort(
        "concurrent_learning_" + std::to_string(i), 4).get_index();
  }

  // Cable direction: [qx, qy, qz]
  cable_direction_ports_.resize(num_drones);
  for (int i = 0; i < num_drones; ++i) {
    cable_direction_ports_[i] = this->DeclareVectorInputPort(
        "cable_direction_" + std::to_string(i), 3).get_index();
  }

  // Wind disturbance: [wx, wy, wz]
  wind_port_ = this->DeclareVectorInputPort("wind", 3).get_index();

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
  if (params_.log_imu_measurements) {
    imu_file_.open(output_dir_ + "/imu_measurements.csv");
  }
  if (params_.log_barometer_measurements) {
    barometer_file_.open(output_dir_ + "/barometer_measurements.csv");
  }
  if (params_.log_rope_states) {
    rope_state_file_.open(output_dir_ + "/rope_states.csv");
  }
  if (params_.log_attitude_data) {
    attitude_file_.open(output_dir_ + "/attitude_data.csv");
  }
  if (params_.log_gpac_signals) {
    gpac_file_.open(output_dir_ + "/gpac_signals.csv");
  }
  if (params_.log_wind_disturbance) {
    wind_file_.open(output_dir_ + "/wind_disturbance.csv");
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
    trajectory_file_ << ",load_qw,load_qx,load_qy,load_qz";
    trajectory_file_ << ",load_wx,load_wy,load_wz";
    for (int i = 0; i < num_drones; ++i) {
      trajectory_file_ << ",drone" << i << "_x,drone" << i << "_y,drone" << i << "_z";
      trajectory_file_ << ",drone" << i << "_vx,drone" << i << "_vy,drone" << i << "_vz";
      trajectory_file_ << ",drone" << i << "_qw,drone" << i << "_qx,drone" << i << "_qy,drone" << i << "_qz";
      trajectory_file_ << ",drone" << i << "_wx,drone" << i << "_wy,drone" << i << "_wz";
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

  // IMU file header
  if (imu_file_.is_open()) {
    imu_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      imu_file_ << ",drone" << i << "_ax,drone" << i << "_ay,drone" << i << "_az";
      imu_file_ << ",drone" << i << "_wx,drone" << i << "_wy,drone" << i << "_wz";
    }
    imu_file_ << "\n";
  }

  // Barometer file header
  if (barometer_file_.is_open()) {
    barometer_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      barometer_file_ << ",drone" << i << "_altitude";
    }
    barometer_file_ << "\n";
  }

  // Rope state file header
  if (rope_state_file_.is_open()) {
    rope_state_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      rope_state_file_ << ",rope" << i << "_stretch";
      rope_state_file_ << ",rope" << i << "_dir_x,rope" << i << "_dir_y,rope" << i << "_dir_z";
      rope_state_file_ << ",rope" << i << "_tension_rate";
    }
    rope_state_file_ << "\n";
  }

  // Attitude file header
  if (attitude_file_.is_open()) {
    attitude_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      // Actual attitude (extracted from trajectory)
      attitude_file_ << ",drone" << i << "_roll,drone" << i << "_pitch,drone" << i << "_yaw";
      // Desired attitude
      attitude_file_ << ",drone" << i << "_des_qw,drone" << i << "_des_qx";
      attitude_file_ << ",drone" << i << "_des_qy,drone" << i << "_des_qz";
      // Attitude error
      attitude_file_ << ",drone" << i << "_err_x,drone" << i << "_err_y,drone" << i << "_err_z";
    }
    attitude_file_ << "\n";
  }

  // GPAC signals file header (ESO, CBF, anti-swing, concurrent learning)
  if (gpac_file_.is_open()) {
    gpac_file_ << "time";
    for (int i = 0; i < num_drones; ++i) {
      // ESO disturbance estimates
      gpac_file_ << ",drone" << i << "_eso_dx,drone" << i << "_eso_dy,drone" << i << "_eso_dz";
      // CBF barrier values
      gpac_file_ << ",drone" << i << "_cbf_taut_lo,drone" << i << "_cbf_taut_hi";
      gpac_file_ << ",drone" << i << "_cbf_angle,drone" << i << "_cbf_swing";
      gpac_file_ << ",drone" << i << "_cbf_tilt,drone" << i << "_cbf_collision";
      // Anti-swing force
      gpac_file_ << ",drone" << i << "_antiswing_fx,drone" << i << "_antiswing_fy,drone" << i << "_antiswing_fz";
      // Cable direction
      gpac_file_ << ",drone" << i << "_cable_qx,drone" << i << "_cable_qy,drone" << i << "_cable_qz";
      // Concurrent learning (parameters + mass estimate)
      gpac_file_ << ",drone" << i << "_cl_theta1,drone" << i << "_cl_theta2";
      gpac_file_ << ",drone" << i << "_cl_theta3,drone" << i << "_cl_mass";
    }
    gpac_file_ << "\n";
  }

  // Wind disturbance file header
  if (wind_file_.is_open()) {
    wind_file_ << "time,wind_vx,wind_vy,wind_vz\n";
  }

  // Flush all headers immediately
  if (trajectory_file_.is_open()) trajectory_file_.flush();
  if (tension_file_.is_open()) tension_file_.flush();
  if (control_file_.is_open()) control_file_.flush();
  if (gps_file_.is_open()) gps_file_.flush();
  if (estimator_file_.is_open()) estimator_file_.flush();
  if (reference_file_.is_open()) reference_file_.flush();
  if (imu_file_.is_open()) imu_file_.flush();
  if (barometer_file_.is_open()) barometer_file_.flush();
  if (rope_state_file_.is_open()) rope_state_file_.flush();
  if (attitude_file_.is_open()) attitude_file_.flush();
  if (gpac_file_.is_open()) gpac_file_.flush();
  if (wind_file_.is_open()) wind_file_.flush();
}

void SimulationDataLogger::ExtractBodyState(
    const Context<double>& plant_context,
    BodyIndex body_index,
    Eigen::Vector3d* position,
    Eigen::Vector3d* velocity,
    Eigen::Quaterniond* orientation,
    Eigen::Vector3d* angular_velocity) const {
  const auto& body = plant_.get_body(body_index);
  RigidTransformd X_WB = plant_.EvalBodyPoseInWorld(plant_context, body);
  *position = X_WB.translation();

  SpatialVelocity<double> V_WB = plant_.EvalBodySpatialVelocityInWorld(
      plant_context, body);
  *velocity = V_WB.translational();

  if (orientation != nullptr) {
    *orientation = X_WB.rotation().ToQuaternion();
  }
  if (angular_velocity != nullptr) {
    *angular_velocity = V_WB.rotational();
  }
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

  // === Log trajectories (ground truth with full pose) ===
  if (trajectory_file_.is_open()) {
    trajectory_file_ << std::fixed << std::setprecision(6) << time;

    // Load state (position, velocity, orientation, angular velocity)
    Eigen::Vector3d load_pos, load_vel, load_omega;
    Eigen::Quaterniond load_quat;
    ExtractBodyState(*plant_context, load_body_index_, &load_pos, &load_vel,
                     &load_quat, &load_omega);
    trajectory_file_ << "," << load_pos.x() << "," << load_pos.y() << "," << load_pos.z();
    trajectory_file_ << "," << load_vel.x() << "," << load_vel.y() << "," << load_vel.z();
    trajectory_file_ << "," << load_quat.w() << "," << load_quat.x()
                     << "," << load_quat.y() << "," << load_quat.z();
    trajectory_file_ << "," << load_omega.x() << "," << load_omega.y() << "," << load_omega.z();

    // Drone states with full pose
    for (int i = 0; i < num_drones; ++i) {
      Eigen::Vector3d drone_pos, drone_vel, drone_omega;
      Eigen::Quaterniond drone_quat;
      ExtractBodyState(*plant_context, drone_body_indices_[i], &drone_pos, &drone_vel,
                       &drone_quat, &drone_omega);
      trajectory_file_ << "," << drone_pos.x() << "," << drone_pos.y() << "," << drone_pos.z();
      trajectory_file_ << "," << drone_vel.x() << "," << drone_vel.y() << "," << drone_vel.z();
      trajectory_file_ << "," << drone_quat.w() << "," << drone_quat.x()
                       << "," << drone_quat.y() << "," << drone_quat.z();
      trajectory_file_ << "," << drone_omega.x() << "," << drone_omega.y() << "," << drone_omega.z();
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

  // === Log IMU measurements ===
  if (imu_file_.is_open()) {
    imu_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(imu_ports_[i]).HasValue(context)) {
        const auto& imu = get_imu_input(i).Eval(context);
        imu_file_ << "," << imu(0) << "," << imu(1) << "," << imu(2);
        imu_file_ << "," << imu(3) << "," << imu(4) << "," << imu(5);
      } else {
        imu_file_ << ",0,0,0,0,0,0";
      }
    }
    imu_file_ << "\n";
  }

  // === Log barometer measurements ===
  if (barometer_file_.is_open()) {
    barometer_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(barometer_ports_[i]).HasValue(context)) {
        const auto& baro = get_barometer_input(i).Eval(context);
        barometer_file_ << "," << baro(0);
      } else {
        barometer_file_ << ",0";
      }
    }
    barometer_file_ << "\n";
  }

  // === Log rope states ===
  if (rope_state_file_.is_open()) {
    rope_state_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      if (get_input_port(rope_state_ports_[i]).HasValue(context)) {
        const auto& rope = get_rope_state_input(i).Eval(context);
        rope_state_file_ << "," << rope(0);  // stretch
        rope_state_file_ << "," << rope(1) << "," << rope(2) << "," << rope(3);  // direction
        rope_state_file_ << "," << rope(4);  // tension rate
      } else {
        rope_state_file_ << ",0,0,0,0,0";
      }
    }
    rope_state_file_ << "\n";
  }

  // === Log attitude data ===
  if (attitude_file_.is_open()) {
    attitude_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      // Get actual attitude from trajectory (already extracted above)
      Eigen::Vector3d drone_pos, drone_vel;
      Eigen::Quaterniond drone_quat;
      ExtractBodyState(*plant_context, drone_body_indices_[i], &drone_pos, &drone_vel, &drone_quat);

      // Convert quaternion to Euler angles (roll, pitch, yaw)
      RotationMatrixd R(drone_quat);
      Eigen::Vector3d rpy = R.IsNearlyEqualTo(RotationMatrixd(), 1e-10) ?
                            Eigen::Vector3d::Zero() : R.IsValid() ?
                            // Use Drake's RollPitchYaw (intrinsic xyz)
                            drake::math::RollPitchYaw<double>(R).vector() : Eigen::Vector3d::Zero();
      attitude_file_ << "," << rpy(0) << "," << rpy(1) << "," << rpy(2);

      // Desired attitude
      if (get_input_port(desired_attitude_ports_[i]).HasValue(context)) {
        const auto& des_att = get_desired_attitude_input(i).Eval(context);
        attitude_file_ << "," << des_att(0) << "," << des_att(1)
                       << "," << des_att(2) << "," << des_att(3);
      } else {
        attitude_file_ << ",1,0,0,0";  // identity quaternion
      }

      // Attitude error
      if (get_input_port(attitude_error_ports_[i]).HasValue(context)) {
        const auto& att_err = get_attitude_error_input(i).Eval(context);
        attitude_file_ << "," << att_err(0) << "," << att_err(1) << "," << att_err(2);
      } else {
        attitude_file_ << ",0,0,0";
      }
    }
    attitude_file_ << "\n";
  }

  // === Log GPAC signals (ESO, CBF, anti-swing, cable direction, concurrent learning) ===
  if (gpac_file_.is_open()) {
    gpac_file_ << std::fixed << std::setprecision(6) << time;
    for (int i = 0; i < num_drones; ++i) {
      // ESO disturbance
      if (get_input_port(eso_disturbance_ports_[i]).HasValue(context)) {
        const auto& eso = get_eso_disturbance_input(i).Eval(context);
        gpac_file_ << "," << eso(0) << "," << eso(1) << "," << eso(2);
      } else {
        gpac_file_ << ",0,0,0";
      }

      // CBF barriers
      if (get_input_port(cbf_barrier_ports_[i]).HasValue(context)) {
        const auto& cbf = get_cbf_barriers_input(i).Eval(context);
        gpac_file_ << "," << cbf(0) << "," << cbf(1) << "," << cbf(2)
                   << "," << cbf(3) << "," << cbf(4) << "," << cbf(5);
      } else {
        gpac_file_ << ",0,0,0,0,0,0";
      }

      // Anti-swing force
      if (get_input_port(antiswing_force_ports_[i]).HasValue(context)) {
        const auto& asw = get_antiswing_force_input(i).Eval(context);
        gpac_file_ << "," << asw(0) << "," << asw(1) << "," << asw(2);
      } else {
        gpac_file_ << ",0,0,0";
      }

      // Cable direction
      if (get_input_port(cable_direction_ports_[i]).HasValue(context)) {
        const auto& cbl = get_cable_direction_input(i).Eval(context);
        gpac_file_ << "," << cbl(0) << "," << cbl(1) << "," << cbl(2);
      } else {
        gpac_file_ << ",0,0,-1";  // default pointing down
      }

      // Concurrent learning
      if (get_input_port(concurrent_learning_ports_[i]).HasValue(context)) {
        const auto& cl = get_concurrent_learning_input(i).Eval(context);
        gpac_file_ << "," << cl(0) << "," << cl(1) << "," << cl(2) << "," << cl(3);
      } else {
        gpac_file_ << ",0,0,0,0";
      }
    }
    gpac_file_ << "\n";
  }

  // === Log wind disturbance ===
  if (wind_file_.is_open()) {
    wind_file_ << std::fixed << std::setprecision(6) << time;
    if (get_input_port(wind_port_).HasValue(context)) {
      const auto& wind = get_wind_input().Eval(context);
      wind_file_ << "," << wind(0) << "," << wind(1) << "," << wind(2);
    } else {
      wind_file_ << ",0,0,0";
    }
    wind_file_ << "\n";
  }

  // Flush all log files periodically to avoid data loss on unexpected termination
  static int flush_counter = 0;
  if (++flush_counter >= 10) {  // Flush every 10 samples
    flush_counter = 0;
    if (trajectory_file_.is_open()) trajectory_file_.flush();
    if (tension_file_.is_open()) tension_file_.flush();
    if (control_file_.is_open()) control_file_.flush();
    if (gps_file_.is_open()) gps_file_.flush();
    if (estimator_file_.is_open()) estimator_file_.flush();
    if (reference_file_.is_open()) reference_file_.flush();
    if (imu_file_.is_open()) imu_file_.flush();
    if (barometer_file_.is_open()) barometer_file_.flush();
    if (rope_state_file_.is_open()) rope_state_file_.flush();
    if (attitude_file_.is_open()) attitude_file_.flush();
    if (gpac_file_.is_open()) gpac_file_.flush();
    if (wind_file_.is_open()) wind_file_.flush();
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

  // Close all basic log files
  if (trajectory_file_.is_open()) trajectory_file_.close();
  if (tension_file_.is_open()) tension_file_.close();
  if (control_file_.is_open()) control_file_.close();
  if (gps_file_.is_open()) gps_file_.close();
  if (estimator_file_.is_open()) estimator_file_.close();
  if (reference_file_.is_open()) reference_file_.close();

  // Close all extended log files
  if (imu_file_.is_open()) imu_file_.close();
  if (barometer_file_.is_open()) barometer_file_.close();
  if (rope_state_file_.is_open()) rope_state_file_.close();
  if (attitude_file_.is_open()) attitude_file_.close();
  if (gpac_file_.is_open()) gpac_file_.close();
  if (wind_file_.is_open()) wind_file_.close();

  finalized_ = true;
  std::cout << "Data logging finalized. Files saved to: " << output_dir_ << std::endl;
}

}  // namespace tether_lift
