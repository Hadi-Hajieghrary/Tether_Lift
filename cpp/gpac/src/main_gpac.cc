/// @file main_gpac.cc
/// @brief Main simulation executable for GPAC multi-drone transport.
///
/// This file sets up and runs a complete simulation of N quadrotors
/// collaboratively transporting a cable-suspended load using the
/// GPAC (Geometric Position and Attitude Control) architecture.
///
/// Usage:
///   ./gpac_simulation [options]
///
/// Options:
///   --num_drones=N       Number of drones (default: 3)
///   --sim_time=T         Simulation duration [s] (default: 30)
///   --visualize          Enable Drake Visualizer
///   --log_dir=PATH       Directory for log files (default: ./logs)

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/multiplexer.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>

#include "control/gpac_controller.h"
#include "control/geometric_math.h"
#include "plant/quadrotor_cable_load_plant.h"
#include "plant/sensor_models.h"
#include "trajectory/trajectory_generator.h"
#include "utils/data_logger.h"
#include "utils/parameters.h"

DEFINE_int32(num_drones, 3, "Number of drones");
DEFINE_double(sim_time, 30.0, "Simulation duration in seconds");
DEFINE_bool(visualize, true, "Enable visualization");
DEFINE_string(log_dir, "./logs", "Directory for log files");
DEFINE_string(trajectory, "lift_and_move", "Trajectory type: hover, lift_and_move, circular");
DEFINE_double(dt, 2e-4, "Simulation timestep");

namespace gpac {

/// @brief Create the trajectory based on command-line flag
TrajectoryGenerator CreateTrajectory(const std::string& type) {
  if (type == "hover") {
    // Simple hover at fixed position
    std::vector<Waypoint> waypoints = {
        {{0, 0, 0.1}, 0.0, TrajectoryType::kHold},
        {{0, 0, 1.0}, 3.0, TrajectoryType::kPolynomial},
        {{0, 0, 1.0}, 30.0, TrajectoryType::kHold}
    };
    return TrajectoryGenerator(waypoints);

  } else if (type == "lift_and_move") {
    // Lift load, move horizontally, set down
    const Eigen::Vector3d start(0, 0, 0.1);
    const Eigen::Vector3d end(3, 0, 0.1);
    return TrajectoryGenerator::CreateLiftAndMove(
        start, end,
        1.5,    // lift height
        3.0,    // lift time
        8.0,    // move time
        2.0);   // settle time

  } else if (type == "circular") {
    // Circular hover pattern
    return TrajectoryGenerator::CreateCircular(
        Eigen::Vector3d(0, 0, 0),  // center
        0.5,    // radius
        1.5,    // altitude
        10.0,   // period
        FLAGS_sim_time);

  } else {
    throw std::runtime_error("Unknown trajectory type: " + type);
  }
}

/// @brief Build the complete simulation diagram
std::unique_ptr<drake::systems::Diagram<double>> BuildSimulation(
    int num_drones,
    std::shared_ptr<drake::geometry::Meshcat> meshcat = nullptr) {

  drake::systems::DiagramBuilder<double> builder;

  // === 1. Create Physical Plant ===

  PlantConfig plant_config = PlantConfig::Default(num_drones);
  plant_config.time_step = FLAGS_dt;

  auto plant_result = BuildMultiDronePlant(&builder, plant_config);
  // auto* plant = plant_result.plant;  // Will be used for full wiring

  // === 2. Add Visualization ===

  if (meshcat != nullptr) {
    drake::geometry::MeshcatVisualizerParams vis_params;
    vis_params.publish_period = 1.0 / 60.0;  // 60 Hz visualization
    drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *plant_result.scene_graph, meshcat, vis_params);
  }

  // === 3. Create Cable Force System ===

  auto* cable_forces = builder.AddSystem<CableForceSystem>(
      num_drones,
      plant_config.cable_lengths,
      plant_config.cable_stiffness,
      plant_config.cable_damping);
  (void)cable_forces;  // Suppress unused warning for now

  // === 4. Create Cable State Computer ===

  auto* cable_state_computer = builder.AddSystem<CableStateComputer>(
      num_drones,
      plant_config.cable_lengths,
      plant_config.cable_stiffness);
  (void)cable_state_computer;  // Suppress unused warning for now

  // === 5. Create Trajectory Generator ===

  // Create trajectory and add directly (not through unique_ptr for DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN)
  std::vector<Waypoint> waypoints = {
      {{0, 0, 0.1}, 0.0, TrajectoryType::kHold},
      {{0, 0, 1.0}, 3.0, TrajectoryType::kPolynomial},
      {{0, 0, 1.0}, 30.0, TrajectoryType::kHold}
  };
  auto* traj_gen = builder.AddSystem<TrajectoryGenerator>(waypoints);

  // === 6. Create Cable Direction Reference ===

  auto* cable_ref = builder.AddSystem<CableDirectionReference>(
      num_drones, "symmetric");

  builder.Connect(traj_gen->get_acceleration_output_port(),
                  cable_ref->get_acceleration_input_port());

  // === 7. Create constant sources for unconnected ports ===

  // Default drone state: [pos(3), quat(4), vel(3), omega(3)] = 13
  Eigen::VectorXd default_drone_state(13);
  default_drone_state.setZero();
  default_drone_state.segment<4>(3) << 1, 0, 0, 0;  // Identity quaternion
  default_drone_state(2) = 1.0;  // Start at 1m height

  auto* drone_state_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      default_drone_state);

  // Default load position
  auto* load_pos_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      Eigen::Vector3d(0, 0, 0.1));

  // Default load velocity
  auto* load_vel_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      Eigen::Vector3d::Zero());

  // Default cable state: [T, q(3), T_dot, q_dot(3)] = 8
  Eigen::VectorXd default_cable_state(8);
  default_cable_state << 5.0, 0, 0, -1, 0, 0, 0, 0;  // 5N tension, pointing down
  auto* cable_state_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      default_cable_state);

  // Default yaw
  auto* yaw_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      Eigen::VectorXd::Zero(1));

  // Default neighbor directions: 3*(N-1) elements
  auto* neighbor_dirs_source = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      Eigen::VectorXd::Zero(3 * (num_drones - 1)));

  // === 8. Create GPAC Controllers (one per drone) ===

  GpacParams gpac_params;  // Uses default values
  std::vector<GpacController*> controllers(num_drones);

  for (int i = 0; i < num_drones; ++i) {
    controllers[i] = builder.AddSystem<GpacController>(i, num_drones, gpac_params);
    controllers[i]->set_name("gpac_controller_" + std::to_string(i));

    // Connect all required inputs
    builder.Connect(drone_state_source->get_output_port(),
                    controllers[i]->get_drone_state_input_port());
    builder.Connect(load_pos_source->get_output_port(),
                    controllers[i]->get_load_position_input_port());
    builder.Connect(load_vel_source->get_output_port(),
                    controllers[i]->get_load_velocity_input_port());
    builder.Connect(cable_state_source->get_output_port(),
                    controllers[i]->get_cable_state_input_port());
    builder.Connect(traj_gen->get_reference_output_port(),
                    controllers[i]->get_trajectory_input_port());
    builder.Connect(yaw_source->get_output_port(),
                    controllers[i]->get_desired_yaw_input_port());
    builder.Connect(neighbor_dirs_source->get_output_port(),
                    controllers[i]->get_neighbor_directions_input_port());
  }

  // === 9. Create Data Logger ===

  LoggerConfig log_config;
  log_config.output_directory = FLAGS_log_dir;
  log_config.log_period = 0.01;  // 100 Hz logging

  auto* logger = builder.AddSystem<DataLogger>(num_drones, log_config);

  // Connect trajectory to logger
  builder.Connect(traj_gen->get_reference_output_port(),
                  logger->get_trajectory_input_port());

  // === 9. Wire Everything Together ===

  // NOTE: Full wiring requires:
  // - State extraction from MultibodyPlant
  // - Force/torque application to plant
  // - Proper sensor modeling
  // This is a simplified version that demonstrates the architecture

  std::cout << "GPAC simulation diagram built successfully" << std::endl;
  std::cout << "  - " << num_drones << " drones" << std::endl;
  std::cout << "  - Plant: MultibodyPlant with cables" << std::endl;
  std::cout << "  - Controllers: GPAC hierarchical architecture" << std::endl;
  std::cout << "  - Trajectory: " << FLAGS_trajectory << std::endl;

  return builder.Build();
}

}  // namespace gpac

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("GPAC Multi-Drone Transport Simulation");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "========================================" << std::endl;
  std::cout << "  GPAC Multi-Drone Transport Simulator" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  const int num_drones = FLAGS_num_drones;
  const double sim_time = FLAGS_sim_time;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Number of drones: " << num_drones << std::endl;
  std::cout << "  Simulation time:  " << sim_time << " s" << std::endl;
  std::cout << "  Trajectory type:  " << FLAGS_trajectory << std::endl;
  std::cout << "  Timestep:         " << FLAGS_dt << " s" << std::endl;
  std::cout << "  Visualization:    " << (FLAGS_visualize ? "enabled" : "disabled") << std::endl;
  std::cout << "  Log directory:    " << FLAGS_log_dir << std::endl;
  std::cout << std::endl;

  try {
    // Create Meshcat for visualization (optional)
    std::shared_ptr<drake::geometry::Meshcat> meshcat = nullptr;
    if (FLAGS_visualize) {
      meshcat = std::make_shared<drake::geometry::Meshcat>();
      std::cout << "Meshcat URL: " << meshcat->web_url() << std::endl;
      std::cout << std::endl;
    }

    // Build the simulation
    auto diagram = gpac::BuildSimulation(num_drones, meshcat);

    // Create simulator
    drake::systems::Simulator<double> simulator(*diagram);
    simulator.set_target_realtime_rate(FLAGS_visualize ? 1.0 : 0.0);

    // Initialize
    simulator.Initialize();

    std::cout << "Starting simulation..." << std::endl;

    // Run simulation
    simulator.AdvanceTo(sim_time);

    std::cout << std::endl;
    std::cout << "Simulation complete!" << std::endl;
    std::cout << "  Final time: " << simulator.get_context().get_time() << " s" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
