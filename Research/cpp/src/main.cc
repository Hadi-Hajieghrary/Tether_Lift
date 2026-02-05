/// @file main.cc
/// Drake simulation: Multiple quadcopters lift a payload using flexible ropes/tethers.
///
/// === Physical Model ===
/// - Quadcopters: N free-floating 6-DoF rigid bodies with thrust and torque control.
/// - Payload: A spherical rigid body resting on the ground with friction.
/// - Ropes/Tethers: N bead-chains, each with tension-only spring-damper segments.
///
/// === Control Strategy ===
/// Each quadcopter uses a cascaded controller:
/// 1. Position controller: PD control to track x/y/z trajectory with formation offset.
/// 2. Attitude controller: PD control to track desired roll/pitch/yaw.
/// 3. Tension-aware pickup: Smooth load transfer using rope tension feedback.
///
/// All quadcopters follow the same trajectory with individual formation offsets.
///
/// === Usage ===
///   ./quad_rope_lift
///
/// The simulation opens a Meshcat visualizer in your browser at http://localhost:7000

#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <random>

#include <Eigen/Core>

#include <drake/common/find_resource.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/externally_applied_spatial_force_multiplexer.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/multibody/tree/spatial_inertia.h>
#include <drake/multibody/tree/unit_inertia.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <drake/systems/primitives/vector_log_sink.h>
#include <drake/systems/primitives/multiplexer.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/systems/primitives/constant_vector_source.h>

#include "quadcopter_controller.h"
#include "rope_force_system.h"
#include "rope_utils.h"
#include "rope_visualizer.h"
#include "tension_plotter.h"
#include "gps_sensor.h"
#include "imu_sensor.h"
#include "barometer_sensor.h"
#include "wind_disturbance.h"
#include "position_velocity_estimator.h"
#include "decentralized_load_estimator.h"
#include "estimation_utils.h"
#include "estimation_error_computer.h"
#include "trajectory_visualizer.h"
#include "simulation_data_logger.h"

namespace quad_rope_lift
{

    using drake::geometry::Box;
    using drake::geometry::HalfSpace;
    using drake::geometry::Meshcat;
    using drake::geometry::MeshcatVisualizer;
    using drake::geometry::ProximityProperties;
    using drake::geometry::Rgba;
    using drake::geometry::SceneGraph;
    using drake::geometry::Sphere;
    using drake::math::RigidTransformd;
    using drake::math::RollPitchYawd;
    using drake::multibody::ContactModel;
    using drake::multibody::CoulombFriction;
    using drake::multibody::ExternallyAppliedSpatialForceMultiplexer;
    using drake::multibody::ModelInstanceIndex;
    using drake::multibody::MultibodyPlant;
    using drake::multibody::Parser;
    using drake::multibody::RigidBody;
    using drake::multibody::SpatialInertia;
    using drake::multibody::UnitInertia;
    using drake::systems::DiagramBuilder;
    using drake::systems::Simulator;
    using drake::systems::VectorLogSink;
    using drake::systems::ZeroOrderHold;
    using drake::systems::Multiplexer;
    using drake::systems::Demultiplexer;
    using drake::systems::ConstantVectorSource;

    /// Configuration for a single quadcopter in the formation.
    struct QuadConfig {
        Eigen::Vector3d initial_position;     ///< Initial (x, y, z) position [m]
        Eigen::Vector3d formation_offset;     ///< Offset from shared trajectory [m]
        Eigen::Vector3d payload_attachment;   ///< Attachment point on payload surface [m]
        double rope_length_mean;              ///< Expected rope rest length [m]
        double rope_length_stddev;            ///< Standard deviation of rope length measurement [m]
        double rope_length;                   ///< Actual rope rest length (sampled) [m]
    };

    int DoMain()
    {
        // =========================================================================
        // Simulation Parameters
        // =========================================================================

        // Physics time step (smaller = more accurate but slower)
        const double simulation_time_step = 2e-4; // [s]

        // Total simulation duration (extended for figure-8 trajectory)
        const double simulation_duration = 50.0; // [s]

        // =========================================================================
        // State Estimation Configuration
        // =========================================================================

        // Enable GPS-based state estimation (false = use ground truth)
        const bool enable_estimation = true;

        // Use estimated state in controller (false = run estimator but use ground truth in controller)
        // Set to false to test estimator accuracy without feedback effects
        const bool use_estimated_in_controller = false;

        // GPS sensor parameters
        const double gps_sample_period = 0.1;  // [s] (10 Hz)
        const Eigen::Vector3d gps_position_noise(0.02, 0.02, 0.05);  // [m] stddev

        // Estimator update rate (should match or be faster than GPS)
        const double estimator_dt = 0.01;  // [s] (100 Hz)

        // =========================================================================
        // Physical Properties
        // =========================================================================

        // Quadcopter (same for all)
        const double quadcopter_mass = 1.5;                            // [kg]
        const Eigen::Vector3d quadcopter_dimensions(0.30, 0.30, 0.10); // [m]

        // Payload
        const double payload_mass = 3.0;    // [kg] - heavier for multi-quad
        const double payload_radius = 0.15; // [m]

        // Rope (common parameters)
        const double rope_total_mass = 0.2;   // [kg] per rope
        const int num_rope_beads = 8;

        const double gravity = 9.81;

        // =========================================================================
        // Multi-Quadcopter Configuration
        // =========================================================================

        // Number of quadcopters (2, 3, or 4 recommended)
        const int num_quadcopters = 3;

        // Initial hover altitude (before ascent)
        const double initial_altitude = 1.2;  // [m]

        // Configure each quadcopter's position and rope attachment
        std::vector<QuadConfig> quad_configs(num_quadcopters);

        // Formation geometry: arrange quadcopters in a circle around payload
        const double formation_radius = 0.6;  // [m] - horizontal distance from payload center
        const double attachment_radius = payload_radius * 0.7;  // Attachment points on payload top

        // =========================================================================
        // Rope Length Distribution Parameters (per quadcopter)
        // Each rope length is sampled from N(mean, stddev^2)
        // =========================================================================

        // Expected rope lengths for each quadcopter [m]
        std::vector<double> rope_length_means = {1.0, 1.1, 0.95};

        // Standard deviations representing measurement uncertainty [m]
        std::vector<double> rope_length_stddevs = {0.05, 0.08, 0.06};

        // Resize if num_quadcopters doesn't match (use last value for extras)
        while (rope_length_means.size() < static_cast<size_t>(num_quadcopters)) {
            rope_length_means.push_back(rope_length_means.back());
            rope_length_stddevs.push_back(rope_length_stddevs.back());
        }

        // Random number generator for Gaussian sampling
        // Use a fixed seed for reproducibility (change seed for different runs)
        const unsigned int random_seed = 42;  // Change this for different random realizations
        std::default_random_engine generator(random_seed);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Rope Length Sampling (Gaussian)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Random seed: " << random_seed << std::endl;

        for (int i = 0; i < num_quadcopters; ++i) {
            const double angle = 2.0 * M_PI * i / num_quadcopters;
            const double x = formation_radius * std::cos(angle);
            const double y = formation_radius * std::sin(angle);

            quad_configs[i].initial_position = Eigen::Vector3d(x, y, initial_altitude);
            quad_configs[i].formation_offset = Eigen::Vector3d(x, y, 0.0);
            quad_configs[i].payload_attachment = Eigen::Vector3d(
                attachment_radius * std::cos(angle),
                attachment_radius * std::sin(angle),
                payload_radius);  // Top of payload

            // Store distribution parameters
            quad_configs[i].rope_length_mean = rope_length_means[i];
            quad_configs[i].rope_length_stddev = rope_length_stddevs[i];

            // Sample rope length from Gaussian distribution
            std::normal_distribution<double> distribution(
                quad_configs[i].rope_length_mean,
                quad_configs[i].rope_length_stddev);
            quad_configs[i].rope_length = distribution(generator);

            // Ensure rope length is positive (re-sample if needed)
            while (quad_configs[i].rope_length <= 0.1) {  // Minimum 10cm
                quad_configs[i].rope_length = distribution(generator);
            }

            std::cout << "Quad " << i << ": mean=" << quad_configs[i].rope_length_mean
                      << "m, stddev=" << quad_configs[i].rope_length_stddev
                      << "m -> sampled=" << quad_configs[i].rope_length << "m" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;

        // =========================================================================
        // Trajectory Waypoints (shared by all quadcopters, plus formation offset)
        // =========================================================================
        // More challenging trajectory: Figure-8 with altitude variation
        // Total duration: ~40 seconds for full maneuver

        std::vector<TrajectoryWaypoint> waypoints;

        // Phase 1: Hover at initial position (stabilize before pickup)
        waypoints.push_back({Eigen::Vector3d(0, 0, initial_altitude), 0.0, 2.0});

        // Phase 2: Ascend to lift payload
        waypoints.push_back({Eigen::Vector3d(0, 0, 3.0), 4.0, 1.5});

        // Phase 3: Begin figure-8 pattern - move to right loop entry
        waypoints.push_back({Eigen::Vector3d(1.5, 0.5, 3.2), 7.0, 0.5});

        // Phase 4: Right loop of figure-8 (clockwise)
        waypoints.push_back({Eigen::Vector3d(2.5, 1.5, 3.5), 10.0, 0.5});  // Top-right
        waypoints.push_back({Eigen::Vector3d(3.0, 0.0, 3.3), 13.0, 0.5});  // Far-right
        waypoints.push_back({Eigen::Vector3d(2.5, -1.5, 3.0), 16.0, 0.5}); // Bottom-right
        waypoints.push_back({Eigen::Vector3d(1.5, -0.5, 2.8), 19.0, 0.5}); // Return to center

        // Phase 5: Cross through center (figure-8 transition)
        waypoints.push_back({Eigen::Vector3d(0.0, 0.0, 3.0), 21.0, 0.5});

        // Phase 6: Left loop of figure-8 (counter-clockwise)
        waypoints.push_back({Eigen::Vector3d(-1.5, 0.5, 3.2), 24.0, 0.5});  // Entry left
        waypoints.push_back({Eigen::Vector3d(-2.5, 1.5, 3.5), 27.0, 0.5});  // Top-left
        waypoints.push_back({Eigen::Vector3d(-3.0, 0.0, 3.3), 30.0, 0.5});  // Far-left
        waypoints.push_back({Eigen::Vector3d(-2.5, -1.5, 3.0), 33.0, 0.5}); // Bottom-left
        waypoints.push_back({Eigen::Vector3d(-1.5, -0.5, 2.8), 36.0, 0.5}); // Return to center

        // Phase 7: Return to origin and descend
        waypoints.push_back({Eigen::Vector3d(0.0, 0.0, 3.0), 39.0, 1.0});

        // Phase 8: Controlled descent to final position
        waypoints.push_back({Eigen::Vector3d(0.0, 0.0, 2.0), 43.0, 2.0});

        // =========================================================================
        // Derived Rope Parameters
        // =========================================================================

        // Calculate segment stiffness to achieve desired maximum stretch under load
        // Each rope carries payload_weight / num_quadcopters
        const double max_stretch_percentage = 0.15; // 15% stretch
        const double load_per_rope = (payload_mass * gravity) / num_quadcopters;

        // Compute average sampled rope length for stiffness calculations
        double avg_rope_length = 0.0;
        for (int i = 0; i < num_quadcopters; ++i) {
            avg_rope_length += quad_configs[i].rope_length;
        }
        avg_rope_length /= num_quadcopters;
        std::cout << "Average sampled rope length: " << avg_rope_length << "m\n" << std::endl;

        const double effective_rope_stiffness =
            load_per_rope / (avg_rope_length * max_stretch_percentage);

        const int num_segments = num_rope_beads + 1;
        const double segment_stiffness = effective_rope_stiffness * num_segments;

        // Segment damping (scaled with sqrt(stiffness))
        const double reference_stiffness = 300.0;
        const double reference_damping = 15.0;
        const double segment_damping =
            reference_damping * std::sqrt(segment_stiffness / reference_stiffness);

        // =========================================================================
        // Build the Simulation
        // =========================================================================

        DiagramBuilder<double> builder;

        // Create the physics engine and scene graph
        auto [plant, scene_graph] =
            drake::multibody::AddMultibodyPlantSceneGraph(&builder, simulation_time_step);
        plant.set_contact_model(ContactModel::kPoint);

        // Model instance for our custom bodies
        const ModelInstanceIndex model_instance =
            plant.AddModelInstance("multi_quad_payload_system");

        // -------------------------------------------------------------------------
        // Create the Quadcopter Bodies
        // -------------------------------------------------------------------------

        const SpatialInertia<double> quad_inertia(
            quadcopter_mass,
            Eigen::Vector3d::Zero(),
            UnitInertia<double>::SolidBox(
                quadcopter_dimensions[0],
                quadcopter_dimensions[1],
                quadcopter_dimensions[2]));

        std::vector<const RigidBody<double>*> quadcopter_bodies;
        std::vector<std::vector<ModelInstanceIndex>> visual_instances_per_quad;
        quadcopter_bodies.reserve(num_quadcopters);
        visual_instances_per_quad.reserve(num_quadcopters);

        Parser parser(&plant);

        for (int i = 0; i < num_quadcopters; ++i) {
            const std::string quad_name = "quadcopter_" + std::to_string(i);

            const RigidBody<double>& quad_body =
                plant.AddRigidBody(quad_name, model_instance, quad_inertia);
            quadcopter_bodies.push_back(&quad_body);

            // Add visual model (Drake's quadrotor mesh) with unique model name
            parser.SetAutoRenaming(true);  // Auto-rename to avoid conflicts
            const std::vector<ModelInstanceIndex> visual_instances =
                parser.AddModels(
                    drake::FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"));

            const RigidBody<double>& visual_base_link =
                plant.GetBodyByName("base_link", visual_instances[0]);
            plant.WeldFrames(
                quad_body.body_frame(),
                visual_base_link.body_frame(),
                RigidTransformd::Identity());

            visual_instances_per_quad.push_back(visual_instances);
        }

        // -------------------------------------------------------------------------
        // Create the Payload Body
        // -------------------------------------------------------------------------

        const SpatialInertia<double> payload_inertia(
            payload_mass,
            Eigen::Vector3d::Zero(),
            UnitInertia<double>::SolidSphere(payload_radius));

        const RigidBody<double>& payload_body =
            plant.AddRigidBody("payload", model_instance, payload_inertia);

        // Add collision and visual geometry
        const CoulombFriction<double> ground_friction(0.5, 0.3);

        plant.RegisterCollisionGeometry(
            payload_body,
            RigidTransformd::Identity(),
            Sphere(payload_radius),
            "payload_collision",
            ground_friction);

        plant.RegisterVisualGeometry(
            payload_body,
            RigidTransformd::Identity(),
            Sphere(payload_radius),
            "payload_visual",
            Eigen::Vector4d(0.8, 0.2, 0.2, 1.0)); // Red

        // -------------------------------------------------------------------------
        // Create the Ground Plane
        // -------------------------------------------------------------------------

        plant.RegisterCollisionGeometry(
            plant.world_body(),
            RigidTransformd::Identity(),
            HalfSpace(),
            "ground_collision",
            ground_friction);

        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransformd(Eigen::Vector3d(0, 0, -0.02)),
            Box(10, 10, 0.04),
            "ground_visual",
            Eigen::Vector4d(0.7, 0.7, 0.7, 1.0)); // Gray

        // -------------------------------------------------------------------------
        // Create the Rope Beads (one chain per quadcopter)
        // -------------------------------------------------------------------------

        // Store bead chains per quadcopter
        std::vector<std::vector<const RigidBody<double>*>> bead_chains(num_quadcopters);
        std::vector<RopeParameters> rope_params_vec(num_quadcopters);

        for (int q = 0; q < num_quadcopters; ++q) {
            rope_params_vec[q] = ComputeRopeParameters(
                num_rope_beads,
                quad_configs[q].rope_length,
                rope_total_mass,
                segment_stiffness,
                segment_damping,
                true,   // bead_diameter_equals_spacing
                0.012); // max_bead_radius

            const SpatialInertia<double> bead_inertia(
                rope_params_vec[q].bead_mass,
                Eigen::Vector3d::Zero(),
                UnitInertia<double>::SolidSphere(rope_params_vec[q].bead_radius));

            bead_chains[q].reserve(num_rope_beads);

            // Color per rope (different hues)
            const double hue = static_cast<double>(q) / num_quadcopters;
            const Eigen::Vector4d bead_color(0.3 + 0.4 * hue, 0.3, 0.3 + 0.4 * (1.0 - hue), 1.0);

            for (int i = 0; i < num_rope_beads; ++i) {
                const std::string name = "rope_" + std::to_string(q) + "_bead_" + std::to_string(i);

                const RigidBody<double>& bead =
                    plant.AddRigidBody(name, model_instance, bead_inertia);

                plant.RegisterCollisionGeometry(
                    bead,
                    RigidTransformd::Identity(),
                    Sphere(rope_params_vec[q].bead_radius),
                    name + "_collision",
                    ground_friction);

                plant.RegisterVisualGeometry(
                    bead,
                    RigidTransformd::Identity(),
                    Sphere(rope_params_vec[q].bead_radius),
                    name + "_visual",
                    bead_color);

                bead_chains[q].push_back(&bead);
            }
        }

        // Finalize the plant
        plant.Finalize();

        // -------------------------------------------------------------------------
        // Define Rope Attachment Points
        // -------------------------------------------------------------------------

        const Eigen::Vector3d quad_attachment_offset(
            0.0, 0.0, -quadcopter_dimensions[2] / 2.0 - 0.05);

        // -------------------------------------------------------------------------
        // Create Rope Force Systems (one per quadcopter)
        // -------------------------------------------------------------------------

        std::vector<RopeForceSystem*> rope_systems;
        std::vector<ZeroOrderHold<double>*> tension_holds;
        rope_systems.reserve(num_quadcopters);
        tension_holds.reserve(num_quadcopters);

        for (int q = 0; q < num_quadcopters; ++q) {
            auto& rope_system = *builder.AddSystem<RopeForceSystem>(
                plant,
                *quadcopter_bodies[q],
                payload_body,
                bead_chains[q],
                quad_attachment_offset,
                quad_configs[q].payload_attachment,
                quad_configs[q].rope_length,
                rope_params_vec[q].segment_stiffness,
                rope_params_vec[q].segment_damping);
            rope_systems.push_back(&rope_system);

            // Zero-order hold for tension signal
            auto& tension_hold = *builder.AddSystem<ZeroOrderHold<double>>(
                simulation_time_step, 4);
            builder.Connect(rope_system.get_tension_output_port(),
                            tension_hold.get_input_port());
            tension_holds.push_back(&tension_hold);
        }

        // -------------------------------------------------------------------------
        // Create Quadcopter Controllers (one per quadcopter)
        // -------------------------------------------------------------------------

        std::vector<QuadcopterLiftController*> controllers;
        controllers.reserve(num_quadcopters);

        for (int q = 0; q < num_quadcopters; ++q) {
            ControllerParams ctrl_params;
            ctrl_params.formation_offset = quad_configs[q].formation_offset;
            ctrl_params.waypoints = waypoints;
            ctrl_params.pickup_target_tension = load_per_rope;  // Share load equally
            ctrl_params.initial_altitude = initial_altitude;
            ctrl_params.final_altitude = 3.0;

            auto& controller = *builder.AddSystem<QuadcopterLiftController>(
                plant, *quadcopter_bodies[q], ctrl_params);
            controllers.push_back(&controller);
        }

        // -------------------------------------------------------------------------
        // Create GPS Sensors and State Estimators (optional)
        // -------------------------------------------------------------------------

        std::vector<GpsSensor*> quad_gps_sensors;
        std::vector<PositionVelocityEstimator*> quad_estimators;
        GpsSensor* load_gps_sensor = nullptr;
        std::vector<DecentralizedLoadEstimator*> load_estimators;
        Multiplexer<double>* load_est_mux = nullptr;  // combines quad 0's pos+vel into 6D

        if (enable_estimation) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "State Estimation Enabled" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "GPS sample rate: " << 1.0/gps_sample_period << " Hz" << std::endl;
            std::cout << "GPS noise (x,y,z): " << gps_position_noise.transpose() << " m" << std::endl;
            std::cout << "Estimator rate: " << 1.0/estimator_dt << " Hz" << std::endl;
            std::cout << "Use estimated in controller: " << (use_estimated_in_controller ? "YES" : "NO") << std::endl;
            std::cout << "========================================\n" << std::endl;

            // GPS sensor parameters
            GpsParams gps_params;
            gps_params.sample_period_sec = gps_sample_period;
            gps_params.position_noise_stddev = gps_position_noise;
            gps_params.dropout_probability = 0.0;  // No dropouts for now

            // Create GPS sensors and estimators for each quadcopter
            quad_gps_sensors.reserve(num_quadcopters);
            quad_estimators.reserve(num_quadcopters);

            for (int q = 0; q < num_quadcopters; ++q) {
                gps_params.random_seed = 100 + q;  // Different seed per quad

                // GPS sensor for quadcopter
                auto& gps = *builder.AddSystem<GpsSensor>(
                    plant, *quadcopter_bodies[q], gps_params);
                quad_gps_sensors.push_back(&gps);

                // State estimator for quadcopter
                EstimatorParams est_params;
                est_params.gps_measurement_noise = gps_position_noise;
                auto& estimator = *builder.AddSystem<PositionVelocityEstimator>(
                    estimator_dt, est_params);
                quad_estimators.push_back(&estimator);

                // Connect GPS to estimator
                builder.Connect(plant.get_state_output_port(),
                                gps.get_plant_state_input_port());
                builder.Connect(gps.get_gps_position_output_port(),
                                estimator.get_gps_position_input_port());
                builder.Connect(gps.get_gps_valid_output_port(),
                                estimator.get_gps_valid_input_port());
            }

            // GPS sensor for load (kept for data logging)
            gps_params.random_seed = 999;
            load_gps_sensor = &(*builder.AddSystem<GpsSensor>(
                plant, payload_body, gps_params));
            builder.Connect(plant.get_state_output_port(),
                            load_gps_sensor->get_plant_state_input_port());

            // Decentralized load estimators (one per quadcopter)
            DecentralizedLoadEstimatorParams dec_est_params;
            load_estimators.reserve(num_quadcopters);

            for (int q = 0; q < num_quadcopters; ++q) {
                // Create per-quad load estimator
                auto& dec_est = *builder.AddSystem<DecentralizedLoadEstimator>(
                    estimator_dt, dec_est_params);
                load_estimators.push_back(&dec_est);

                // Split quad estimator output [x,y,z,vx,vy,vz] into pos(3) + vel(3)
                auto& est_demux = *builder.AddSystem<Demultiplexer<double>>(6, 3);
                builder.Connect(quad_estimators[q]->get_estimated_state_output_port(),
                                est_demux.get_input_port(0));
                builder.Connect(est_demux.get_output_port(0),
                                dec_est.get_quad_position_input_port());
                builder.Connect(est_demux.get_output_port(1),
                                dec_est.get_quad_velocity_input_port());

                // Cable direction from rope tension vector
                auto& cable_dir = *builder.AddSystem<CableDirectionFromTension>();
                builder.Connect(rope_systems[q]->get_tension_output_port(),
                                cable_dir.get_tension_input_port());
                builder.Connect(cable_dir.get_direction_output_port(),
                                dec_est.get_cable_direction_input_port());

                // Cable length (constant scalar)
                Eigen::VectorXd len_vec(1);
                len_vec << quad_configs[q].rope_length;
                auto& cable_len = *builder.AddSystem<ConstantVectorSource<double>>(len_vec);
                builder.Connect(cable_len.get_output_port(),
                                dec_est.get_cable_length_input_port());

                // Scalar tension from 4D tension output [T, fx, fy, fz]
                auto& tension_demux = *builder.AddSystem<Demultiplexer<double>>(
                    std::vector<int>{1, 3});
                builder.Connect(rope_systems[q]->get_tension_output_port(),
                                tension_demux.get_input_port(0));
                builder.Connect(tension_demux.get_output_port(0),
                                dec_est.get_cable_tension_input_port());
            }

            // Combine quad 0's load estimate into 6D [p, v] for downstream consumers
            load_est_mux = &(*builder.AddSystem<Multiplexer<double>>(
                std::vector<int>{3, 3}));
            builder.Connect(load_estimators[0]->get_load_position_output_port(),
                            load_est_mux->get_input_port(0));
            builder.Connect(load_estimators[0]->get_load_velocity_output_port(),
                            load_est_mux->get_input_port(1));
        }

        // -------------------------------------------------------------------------
        // Create IMU Sensors (for all quadcopters)
        // -------------------------------------------------------------------------

        std::vector<ImuSensor*> quad_imu_sensors;
        quad_imu_sensors.reserve(num_quadcopters);

        ImuParams imu_params;
        imu_params.sample_period_sec = 0.005;  // 200 Hz
        imu_params.gyro_noise_density = Eigen::Vector3d(5e-4, 5e-4, 5e-4);
        imu_params.accel_noise_density = Eigen::Vector3d(4e-3, 4e-3, 4e-3);

        for (int q = 0; q < num_quadcopters; ++q) {
            imu_params.random_seed = 200 + q;  // Different seed per quad
            auto& imu = *builder.AddSystem<ImuSensor>(
                plant, *quadcopter_bodies[q], imu_params);
            quad_imu_sensors.push_back(&imu);

            // Connect IMU to plant state
            builder.Connect(plant.get_state_output_port(),
                            imu.get_plant_state_input_port());
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "IMU Sensors Enabled" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "IMU sample rate: " << 1.0/imu_params.sample_period_sec << " Hz" << std::endl;
        std::cout << "Gyro noise density: " << imu_params.gyro_noise_density.transpose() << " rad/s/sqrt(Hz)" << std::endl;
        std::cout << "Accel noise density: " << imu_params.accel_noise_density.transpose() << " m/s^2/sqrt(Hz)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // -------------------------------------------------------------------------
        // Create Barometer Sensors (for all quadcopters)
        // -------------------------------------------------------------------------

        std::vector<BarometerSensor*> quad_barometers;
        quad_barometers.reserve(num_quadcopters);

        BarometerParams baro_params;
        baro_params.sample_period_sec = 0.04;  // 25 Hz
        baro_params.white_noise_stddev = 0.3;
        baro_params.correlated_noise_stddev = 0.2;

        for (int q = 0; q < num_quadcopters; ++q) {
            baro_params.random_seed = 300 + q;  // Different seed per quad
            auto& baro = *builder.AddSystem<BarometerSensor>(
                plant, *quadcopter_bodies[q], baro_params);
            quad_barometers.push_back(&baro);

            // Connect barometer to plant state
            builder.Connect(plant.get_state_output_port(),
                            baro.get_plant_state_input_port());
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Barometer Sensors Enabled" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Barometer sample rate: " << 1.0/baro_params.sample_period_sec << " Hz" << std::endl;
        std::cout << "White noise stddev: " << baro_params.white_noise_stddev << " m" << std::endl;
        std::cout << "Correlated noise stddev: " << baro_params.correlated_noise_stddev << " m" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // -------------------------------------------------------------------------
        // Create Wind Disturbance System
        // -------------------------------------------------------------------------

        DrydenTurbulenceParams wind_params;
        wind_params.mean_wind = Eigen::Vector3d(1.0, 0.5, 0.0);  // Light wind
        wind_params.sigma_u = 0.5;  // Reduced turbulence for controlled sim
        wind_params.sigma_v = 0.5;
        wind_params.sigma_w = 0.25;
        wind_params.altitude_dependent = true;

        GustParams gust_params;
        gust_params.enabled = false;  // Disable gusts for now

        auto& wind_system = *builder.AddSystem<WindDisturbance>(
            num_quadcopters, wind_params, gust_params, 0.01);

        // Create a position extractor for wind system input (drone body centers)
        std::vector<const RigidBody<double>*> wind_quad_bodies;
        std::vector<Eigen::Vector3d> wind_zero_offsets;
        for (int q = 0; q < num_quadcopters; ++q) {
            wind_quad_bodies.push_back(quadcopter_bodies[q]);
            wind_zero_offsets.push_back(Eigen::Vector3d::Zero());
        }
        auto& drone_pos_extractor = *builder.AddSystem<AttachmentPositionExtractor>(
            plant, wind_quad_bodies, wind_zero_offsets);
        builder.Connect(plant.get_state_output_port(),
                        drone_pos_extractor.get_plant_state_input_port());
        builder.Connect(drone_pos_extractor.get_positions_output_port(),
                        wind_system.get_drone_positions_input_port());

        std::cout << "\n========================================" << std::endl;
        std::cout << "Wind Disturbance Enabled" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Mean wind: " << wind_params.mean_wind.transpose() << " m/s" << std::endl;
        std::cout << "Turbulence (u,v,w): " << wind_params.sigma_u << ", "
                  << wind_params.sigma_v << ", " << wind_params.sigma_w << " m/s" << std::endl;
        std::cout << "Gusts enabled: " << (gust_params.enabled ? "YES" : "NO") << std::endl;
        std::cout << "========================================\n" << std::endl;

        // -------------------------------------------------------------------------
        // Connect Systems
        // -------------------------------------------------------------------------

        // Force combiner: 2 inputs per quadcopter (rope + controller)
        auto& force_combiner = *builder.AddSystem<ExternallyAppliedSpatialForceMultiplexer>(
            2 * num_quadcopters);

        for (int q = 0; q < num_quadcopters; ++q) {
            // Rope system connections (always use ground truth for physics)
            builder.Connect(plant.get_state_output_port(),
                            rope_systems[q]->get_plant_state_input_port());
            builder.Connect(rope_systems[q]->get_forces_output_port(),
                            force_combiner.get_input_port(2 * q + 1));

            // Controller connections
            builder.Connect(plant.get_state_output_port(),
                            controllers[q]->get_plant_state_input_port());
            builder.Connect(tension_holds[q]->get_output_port(),
                            controllers[q]->get_tension_input_port());
            builder.Connect(controllers[q]->get_control_output_port(),
                            force_combiner.get_input_port(2 * q));

            // Connect estimated state to controller (if estimation enabled and configured)
            if (enable_estimation && use_estimated_in_controller) {
                builder.Connect(quad_estimators[q]->get_estimated_state_output_port(),
                                controllers[q]->get_estimated_state_input_port());
            }
        }

        // Apply combined forces to plant
        builder.Connect(force_combiner.get_output_port(),
                        plant.get_applied_spatial_force_input_port());

        // Log tension data (just first rope for now)
        auto& tension_logger = *builder.AddSystem<VectorLogSink<double>>(4);
        builder.Connect(rope_systems[0]->get_tension_output_port(),
                        tension_logger.get_input_port());

        // -------------------------------------------------------------------------
        // Estimation Error Logging (if estimation enabled)
        // -------------------------------------------------------------------------

        std::vector<VectorLogSink<double>*> quad_error_loggers;
        VectorLogSink<double>* load_error_logger = nullptr;

        if (enable_estimation) {
            // Log estimation errors for each quadcopter
            for (int q = 0; q < num_quadcopters; ++q) {
                auto& error_computer = *builder.AddSystem<EstimationErrorComputer>(
                    plant, *quadcopter_bodies[q]);
                builder.Connect(plant.get_state_output_port(),
                                error_computer.get_plant_state_input_port());
                builder.Connect(quad_estimators[q]->get_estimated_state_output_port(),
                                error_computer.get_estimated_state_input_port());

                auto& error_logger = *builder.AddSystem<VectorLogSink<double>>(8);
                builder.Connect(error_computer.get_error_output_port(),
                                error_logger.get_input_port());
                quad_error_loggers.push_back(&error_logger);
            }

            // Log estimation error for load
            auto& load_error_computer = *builder.AddSystem<EstimationErrorComputer>(
                plant, payload_body);
            builder.Connect(plant.get_state_output_port(),
                            load_error_computer.get_plant_state_input_port());
            builder.Connect(load_est_mux->get_output_port(0),
                            load_error_computer.get_estimated_state_input_port());

            load_error_logger = &(*builder.AddSystem<VectorLogSink<double>>(8));
            builder.Connect(load_error_computer.get_error_output_port(),
                            load_error_logger->get_input_port());
        }

        // -------------------------------------------------------------------------
        // Set Up Visualization
        // -------------------------------------------------------------------------

        auto meshcat = std::make_shared<Meshcat>();
        meshcat->Delete();
        MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat);

        // Add rope polyline visualizers (one per rope)
        std::vector<RopeVisualizer*> rope_visualizers;
        rope_visualizers.reserve(num_quadcopters);

        // Colors for ropes
        std::vector<Rgba> rope_colors = {
            Rgba(0.2, 0.2, 0.8, 1.0),  // Blue
            Rgba(0.2, 0.8, 0.2, 1.0),  // Green
            Rgba(0.8, 0.8, 0.2, 1.0),  // Yellow
            Rgba(0.8, 0.2, 0.8, 1.0),  // Magenta
        };

        for (int q = 0; q < num_quadcopters; ++q) {
            std::vector<std::pair<const RigidBody<double>*, Eigen::Vector3d>> rope_path_points;
            rope_path_points.reserve(num_rope_beads + 2);

            rope_path_points.emplace_back(quadcopter_bodies[q], quad_attachment_offset);
            for (const auto* bead : bead_chains[q]) {
                rope_path_points.emplace_back(bead, Eigen::Vector3d::Zero());
            }
            rope_path_points.emplace_back(&payload_body, quad_configs[q].payload_attachment);

            const std::string rope_path = "rope_line_" + std::to_string(q);
            const Rgba& color = rope_colors[q % rope_colors.size()];

            auto& rope_visualizer = *builder.AddSystem<RopeVisualizer>(
                plant, rope_path_points, meshcat, rope_path, 3.0, color);
            builder.Connect(plant.get_state_output_port(),
                            rope_visualizer.get_plant_state_input_port());
            rope_visualizers.push_back(&rope_visualizer);
        }

        // -------------------------------------------------------------------------
        // Add Trajectory Visualizer (Reference Path + Actual Trails)
        // -------------------------------------------------------------------------

        // Collect drone body indices for visualization
        std::vector<drake::multibody::BodyIndex> drone_body_indices;
        for (int q = 0; q < num_quadcopters; ++q) {
            drone_body_indices.push_back(quadcopter_bodies[q]->index());
        }

        // Configure trajectory visualization
        tether_lift::TrajectoryVisualizer::Params traj_vis_params;
        traj_vis_params.show_reference_trajectory = true;
        traj_vis_params.reference_color = drake::geometry::Rgba(0.0, 1.0, 0.0, 0.8);  // Green
        traj_vis_params.reference_line_width = 5.0;
        traj_vis_params.show_trails = true;
        traj_vis_params.trail_update_period = 0.05;
        traj_vis_params.load_trail_color = drake::geometry::Rgba(1.0, 0.5, 0.0, 0.9);  // Orange
        traj_vis_params.load_trail_width = 4.0;
        traj_vis_params.load_max_trail_points = 2000;
        traj_vis_params.drone_max_trail_points = 1000;

        auto& trajectory_visualizer = *builder.AddSystem<tether_lift::TrajectoryVisualizer>(
            plant,
            meshcat,
            payload_body.index(),
            drone_body_indices,
            traj_vis_params);
        builder.Connect(plant.get_state_output_port(),
                        trajectory_visualizer.get_plant_state_input());

        // -------------------------------------------------------------------------
        // Add Tension Plotter for Real-Time Visualization
        // -------------------------------------------------------------------------

        auto& tension_plotter = *builder.AddSystem<TensionPlotter>(
            meshcat,
            num_quadcopters,
            rope_colors,
            10.0,   // time_window [s]
            load_per_rope * 2.5,  // max_tension [N] - 2.5x expected for headroom
            0.05);  // update_period [s]

        // Connect tension signals from each rope system to the plotter
        for (int q = 0; q < num_quadcopters; ++q) {
            builder.Connect(rope_systems[q]->get_tension_output_port(),
                            tension_plotter.get_tension_input_port(q));
        }

        // -------------------------------------------------------------------------
        // Add Data Logger for Comprehensive Signal Logging
        // -------------------------------------------------------------------------

        tether_lift::SimulationDataLogger::Params logger_params;
        logger_params.base_output_dir = "/workspaces/Tether_Lift/outputs/logs";
        logger_params.log_period = 0.01;  // 100 Hz logging
        logger_params.num_quadcopters = num_quadcopters;

        // Basic logging (always enabled)
        logger_params.log_plant_state = true;
        logger_params.log_tensions = true;
        logger_params.log_control_efforts = true;
        logger_params.log_gps_measurements = enable_estimation;
        logger_params.log_estimator_outputs = enable_estimation;
        logger_params.log_reference_trajectory = false;  // No separate trajectory system yet

        // Extended logging flags - ALL ENABLED
        logger_params.log_imu_measurements = true;        // IMU sensors enabled
        logger_params.log_barometer_measurements = true;  // Barometer sensors enabled
        logger_params.log_rope_states = false;            // Need rope state output ports (future)
        logger_params.log_attitude_data = true;           // Extract from plant state
        logger_params.log_gpac_signals = false;           // GPAC controllers not used yet
        logger_params.log_wind_disturbance = true;        // Wind system enabled

        auto& data_logger = *builder.AddSystem<tether_lift::SimulationDataLogger>(
            plant,
            payload_body.index(),
            drone_body_indices,
            logger_params);

        // Connect plant state
        builder.Connect(plant.get_state_output_port(),
                        data_logger.get_plant_state_input());

        // Connect tension signals
        for (int q = 0; q < num_quadcopters; ++q) {
            builder.Connect(rope_systems[q]->get_tension_output_port(),
                            data_logger.get_tension_input(q));
        }

        // Connect control efforts (from controllers)
        for (int q = 0; q < num_quadcopters; ++q) {
            builder.Connect(controllers[q]->get_control_vector_output_port(),
                            data_logger.get_control_effort_input(q));
        }

        // Connect GPS and estimator outputs (if estimation enabled)
        if (enable_estimation) {
            for (int q = 0; q < num_quadcopters; ++q) {
                builder.Connect(quad_gps_sensors[q]->get_gps_position_output_port(),
                                data_logger.get_gps_input(q));
                builder.Connect(quad_estimators[q]->get_estimated_state_output_port(),
                                data_logger.get_estimated_state_input(q));
            }
            builder.Connect(load_gps_sensor->get_gps_position_output_port(),
                            data_logger.get_load_gps_input());
            builder.Connect(load_est_mux->get_output_port(0),
                            data_logger.get_load_estimated_state_input());
        }

        // Connect IMU measurements (combined accel + gyro into 6D vector)
        for (int q = 0; q < num_quadcopters; ++q) {
            // Create multiplexer to combine accel (3D) and gyro (3D) into 6D IMU vector
            auto& imu_mux = *builder.AddSystem<Multiplexer<double>>(std::vector<int>{3, 3});
            builder.Connect(quad_imu_sensors[q]->get_accel_output_port(),
                            imu_mux.get_input_port(0));
            builder.Connect(quad_imu_sensors[q]->get_gyro_output_port(),
                            imu_mux.get_input_port(1));
            builder.Connect(imu_mux.get_output_port(0),
                            data_logger.get_imu_input(q));
        }

        // Connect barometer measurements
        for (int q = 0; q < num_quadcopters; ++q) {
            builder.Connect(quad_barometers[q]->get_altitude_output_port(),
                            data_logger.get_barometer_input(q));
        }

        // Connect wind disturbance (use first drone's wind as representative)
        // Wind system outputs [3*N] vector, we need first 3 elements
        auto& wind_demux = *builder.AddSystem<Demultiplexer<double>>(
            3 * num_quadcopters, 3);
        builder.Connect(wind_system.get_wind_velocities_output_port(),
                        wind_demux.get_input_port(0));
        builder.Connect(wind_demux.get_output_port(0),  // First drone's wind
                        data_logger.get_wind_input());

        // -------------------------------------------------------------------------
        // Build and Initialize Simulation
        // -------------------------------------------------------------------------

        auto diagram = builder.Build();
        Simulator<double> simulator(*diagram);
        auto& context = simulator.get_mutable_context();
        auto& plant_context = plant.GetMyMutableContextFromRoot(&context);

        // Update controller masses to include visual model mass
        for (int q = 0; q < num_quadcopters; ++q) {
            const double visual_model_mass =
                plant.CalcTotalMass(plant_context, visual_instances_per_quad[q]);
            controllers[q]->set_mass(quadcopter_mass + visual_model_mass);
        }

        // Set initial poses
        const double ground_clearance = 0.01;

        plant.SetFreeBodyPose(
            &plant_context,
            payload_body,
            RigidTransformd(Eigen::Vector3d(0.0, 0.0, payload_radius + ground_clearance)));

        for (int q = 0; q < num_quadcopters; ++q) {
            plant.SetFreeBodyPose(
                &plant_context,
                *quadcopter_bodies[q],
                RigidTransformd(quad_configs[q].initial_position));
        }

        // Initialize rope beads in slack configuration (per rope)
        for (int q = 0; q < num_quadcopters; ++q) {
            const auto& quad_pose = plant.EvalBodyPoseInWorld(plant_context, *quadcopter_bodies[q]);
            const auto& payload_pose = plant.EvalBodyPoseInWorld(plant_context, payload_body);
            const Eigen::Vector3d rope_start_world = quad_pose * quad_attachment_offset;
            const Eigen::Vector3d rope_end_world = payload_pose * quad_configs[q].payload_attachment;

            // Lateral direction perpendicular to rope for slack
            Eigen::Vector3d lateral = Eigen::Vector3d::UnitZ().cross(
                (rope_end_world - rope_start_world).normalized());
            if (lateral.norm() < 0.1) {
                lateral = Eigen::Vector3d::UnitX();
            }
            lateral.normalize();

            const auto initial_bead_positions = GenerateSlackRopePositions(
                rope_start_world,
                rope_end_world,
                num_rope_beads,
                quad_configs[q].rope_length,
                0.85,       // slack_ratio
                lateral,
                1.0);       // max_lateral_amplitude

            for (int i = 0; i < num_rope_beads; ++i) {
                plant.SetFreeBodyPose(
                    &plant_context,
                    *bead_chains[q][i],
                    RigidTransformd(initial_bead_positions[i]));
            }
        }

        // Initialize state estimators (if enabled)
        if (enable_estimation) {
            // Initialize GPS sensors with initial positions
            for (int q = 0; q < num_quadcopters; ++q) {
                auto& gps_context = quad_gps_sensors[q]->GetMyMutableContextFromRoot(&context);
                quad_gps_sensors[q]->InitializeGpsState(
                    &gps_context,
                    quad_configs[q].initial_position);
            }

            // Initialize load GPS sensor
            auto& load_gps_context = load_gps_sensor->GetMyMutableContextFromRoot(&context);
            load_gps_sensor->InitializeGpsState(
                &load_gps_context,
                Eigen::Vector3d(0.0, 0.0, payload_radius + ground_clearance));

            // Initialize quad estimators with true initial positions
            for (int q = 0; q < num_quadcopters; ++q) {
                auto& est_context = quad_estimators[q]->GetMyMutableContextFromRoot(&context);
                quad_estimators[q]->SetInitialState(
                    &est_context,
                    quad_configs[q].initial_position,
                    Eigen::Vector3d::Zero());
            }

            // Initialize decentralized load estimators with true initial position
            for (int q = 0; q < num_quadcopters; ++q) {
                auto& load_est_context = load_estimators[q]->GetMyMutableContextFromRoot(&context);
                load_estimators[q]->SetInitialState(
                    &load_est_context,
                    Eigen::Vector3d(0.0, 0.0, payload_radius + ground_clearance),
                    Eigen::Vector3d::Zero());
            }
        }

        // Publish initial state
        diagram->ForcedPublish(context);

        // -------------------------------------------------------------------------
        // Draw Reference Trajectory (before simulation starts)
        // -------------------------------------------------------------------------

        // Compute LOAD reference trajectory (not quadcopter trajectory)
        // The waypoints are for quadcopters - the load hangs below by rope length
        // We need to show where the load CENTER should ideally be
        //
        // Vertical distance from drone center to load center:
        //   1. quad_attachment_offset.z() = -(quad_height/2 + 0.05) ≈ -0.10m (below drone)
        //   2. rope_length (with stretch under load ~15%) ≈ rope * 1.15
        //   3. payload_attachment.z() = payload_radius ≈ 0.15m (above load center)
        //
        // So load_center_z = drone_z + quad_attachment_offset.z() - rope_length*1.15 - payload_radius
        //                  = drone_z - 0.10 - rope*1.15 - 0.15
        //                  = drone_z - (0.25 + rope*1.15)

        const double rope_stretch_factor = 1.0 + max_stretch_percentage;  // 1.15
        const double total_vertical_offset =
            std::abs(quad_attachment_offset.z()) +  // Distance from drone center to rope attachment
            avg_rope_length * rope_stretch_factor + // Stretched rope length
            payload_radius;                          // Distance from payload attachment to center

        std::cout << "Reference trajectory offset calculation:" << std::endl;
        std::cout << "  quad_attachment_offset.z = " << quad_attachment_offset.z() << " m" << std::endl;
        std::cout << "  avg_rope_length = " << avg_rope_length << " m" << std::endl;
        std::cout << "  rope_stretch_factor = " << rope_stretch_factor << std::endl;
        std::cout << "  payload_radius = " << payload_radius << " m" << std::endl;
        std::cout << "  total_vertical_offset = " << total_vertical_offset << " m" << std::endl;

        std::vector<Eigen::Vector3d> load_reference_waypoints;
        load_reference_waypoints.reserve(waypoints.size() + 1);

        // Start from load's actual initial position on the ground
        const double load_initial_z = payload_radius + ground_clearance;
        load_reference_waypoints.push_back(Eigen::Vector3d(0, 0, load_initial_z));

        // For subsequent waypoints, compute load position as quad position minus total offset
        for (const auto& wp : waypoints) {
            // Load center position accounting for all offsets
            double load_z = wp.position.z() - total_vertical_offset;

            // Load can't go below ground (load center is at payload_radius above ground)
            load_z = std::max(load_z, load_initial_z);

            // Only add if this creates meaningful movement from previous point
            Eigen::Vector3d load_pos(wp.position.x(), wp.position.y(), load_z);
            if (load_reference_waypoints.empty() ||
                (load_pos - load_reference_waypoints.back()).norm() > 0.05) {
                load_reference_waypoints.push_back(load_pos);
            }
        }

        trajectory_visualizer.DrawReferenceTrajectory(load_reference_waypoints);

        std::cout << "Load reference waypoints:" << std::endl;
        for (size_t i = 0; i < load_reference_waypoints.size(); ++i) {
            std::cout << "  [" << i << "]: (" << load_reference_waypoints[i].x()
                      << ", " << load_reference_waypoints[i].y()
                      << ", " << load_reference_waypoints[i].z() << ")" << std::endl;
        }

        // -------------------------------------------------------------------------
        // Write Configuration File for Data Logger
        // -------------------------------------------------------------------------

        std::map<std::string, std::string> config_params;
        config_params["simulation_time_step"] = std::to_string(simulation_time_step);
        config_params["simulation_duration"] = std::to_string(simulation_duration);
        config_params["num_quadcopters"] = std::to_string(num_quadcopters);
        config_params["quadcopter_mass"] = std::to_string(quadcopter_mass);
        config_params["payload_mass"] = std::to_string(payload_mass);
        config_params["payload_radius"] = std::to_string(payload_radius);
        config_params["initial_altitude"] = std::to_string(initial_altitude);
        config_params["formation_radius"] = std::to_string(formation_radius);
        config_params["avg_rope_length"] = std::to_string(avg_rope_length);
        config_params["segment_stiffness"] = std::to_string(segment_stiffness);
        config_params["segment_damping"] = std::to_string(segment_damping);
        config_params["num_rope_beads"] = std::to_string(num_rope_beads);
        config_params["enable_estimation"] = enable_estimation ? "true" : "false";
        config_params["use_estimated_in_controller"] = use_estimated_in_controller ? "true" : "false";
        config_params["gps_sample_period"] = std::to_string(gps_sample_period);
        config_params["estimator_dt"] = std::to_string(estimator_dt);
        config_params["random_seed"] = std::to_string(random_seed);

        // Log rope lengths for each quadcopter
        for (int q = 0; q < num_quadcopters; ++q) {
            config_params["quad_" + std::to_string(q) + "_rope_length_mean"] =
                std::to_string(quad_configs[q].rope_length_mean);
            config_params["quad_" + std::to_string(q) + "_rope_length_sampled"] =
                std::to_string(quad_configs[q].rope_length);
        }

        // Log waypoints
        for (size_t i = 0; i < waypoints.size(); ++i) {
            config_params["waypoint_" + std::to_string(i) + "_position"] =
                "(" + std::to_string(waypoints[i].position.x()) + ", " +
                std::to_string(waypoints[i].position.y()) + ", " +
                std::to_string(waypoints[i].position.z()) + ")";
            config_params["waypoint_" + std::to_string(i) + "_arrival_time"] =
                std::to_string(waypoints[i].arrival_time);
            config_params["waypoint_" + std::to_string(i) + "_hold_time"] =
                std::to_string(waypoints[i].hold_time);
        }

        data_logger.WriteConfigFile(config_params);

        // -------------------------------------------------------------------------
        // Run Simulation
        // -------------------------------------------------------------------------

        simulator.set_target_realtime_rate(1.0);

        // Enable MeshCat recording for video export
        const bool enable_recording = true;
        if (enable_recording) {
            meshcat->StartRecording();
        }

        std::cout << "Starting multi-quadcopter simulation...\n";
        std::cout << "  Number of quadcopters: " << num_quadcopters << "\n";
        std::cout << "  Payload mass: " << payload_mass << " kg\n";
        std::cout << "  Load per quadcopter: " << load_per_rope / gravity << " kg\n";
        std::cout << "  Duration: " << simulation_duration << " s\n";
        std::cout << "Open Meshcat at: http://localhost:7000\n\n";

        // Run in chunks to show progress
        const double progress_interval = 0.1;
        double current_time = 0.0;

        while (current_time < simulation_duration) {
            current_time += progress_interval;
            simulator.AdvanceTo(current_time);

            // Print progress every second
            if (static_cast<int>(current_time * 10) % 10 == 0) {
                std::cout << "  Simulated " << current_time << "s / "
                          << simulation_duration << "s\n";
            }
        }

        // Stop recording and export
        if (enable_recording) {
            meshcat->StopRecording();
            meshcat->PublishRecording();  // Makes it playable in browser

            // Export to static HTML file
            const std::string html_path = "/workspaces/Tether_Lift/Research/outputs/simulation_recording.html";
            std::ofstream html_file(html_path);
            html_file << meshcat->StaticHtml();
            html_file.close();
            std::cout << "\nSaved MeshCat recording to: " << html_path << "\n";
            std::cout << "Open the HTML file in a browser to replay the simulation.\n";
        }

        // -------------------------------------------------------------------------
        // Print Results
        // -------------------------------------------------------------------------

        const auto& tension_log = tension_logger.FindLog(context);
        const auto& tension_data = tension_log.data();

        double max_tension = 0.0;
        for (int i = 0; i < tension_data.cols(); ++i) {
            max_tension = std::max(max_tension, tension_data(0, i));
        }

        std::cout << "\nSimulation complete!\n";
        std::cout << "  Max rope tension (rope 0): " << max_tension << " N\n";
        std::cout << "  Expected tension per rope: " << load_per_rope << " N\n";
        std::cout << "  Total payload weight: " << payload_mass * gravity << " N\n";

        // Print estimation error statistics (if enabled)
        if (enable_estimation && !quad_error_loggers.empty()) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Estimation Error Statistics" << std::endl;
            std::cout << "========================================" << std::endl;

            for (int q = 0; q < num_quadcopters; ++q) {
                const auto& error_log = quad_error_loggers[q]->FindLog(context);
                const auto& error_data = error_log.data();

                // Compute RMS position error (column 6 is position norm error)
                double pos_err_sum = 0.0;
                double max_pos_err = 0.0;
                for (int i = 0; i < error_data.cols(); ++i) {
                    const double pos_err = error_data(6, i);
                    pos_err_sum += pos_err * pos_err;
                    max_pos_err = std::max(max_pos_err, pos_err);
                }
                const double rms_pos_err = std::sqrt(pos_err_sum / error_data.cols());

                std::cout << "Quad " << q << ": RMS pos error = " << rms_pos_err * 100
                          << " cm, Max = " << max_pos_err * 100 << " cm" << std::endl;
            }

            // Load estimation error
            if (load_error_logger != nullptr) {
                const auto& load_error_log = load_error_logger->FindLog(context);
                const auto& load_error_data = load_error_log.data();

                double pos_err_sum = 0.0;
                double max_pos_err = 0.0;
                for (int i = 0; i < load_error_data.cols(); ++i) {
                    const double pos_err = load_error_data(6, i);
                    pos_err_sum += pos_err * pos_err;
                    max_pos_err = std::max(max_pos_err, pos_err);
                }
                const double rms_pos_err = std::sqrt(pos_err_sum / load_error_data.cols());

                std::cout << "Load:   RMS pos error = " << rms_pos_err * 100
                          << " cm, Max = " << max_pos_err * 100 << " cm" << std::endl;
            }
            std::cout << "========================================\n" << std::endl;
        }

        // Finalize data logging
        data_logger.Finalize();
        std::cout << "Data logged to: " << data_logger.output_dir() << std::endl;

        return 0;
    }

} // namespace quad_rope_lift

int main()
{
    return quad_rope_lift::DoMain();
}
