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
#include <memory>
#include <vector>
#include <string>

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

#include "quadcopter_controller.h"
#include "rope_force_system.h"
#include "rope_utils.h"
#include "rope_visualizer.h"
#include "tension_plotter.h"

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

    /// Configuration for a single quadcopter in the formation.
    struct QuadConfig {
        Eigen::Vector3d initial_position;     ///< Initial (x, y, z) position [m]
        Eigen::Vector3d formation_offset;     ///< Offset from shared trajectory [m]
        Eigen::Vector3d payload_attachment;   ///< Attachment point on payload surface [m]
        double rope_length;                   ///< Rope rest length [m]
    };

    int DoMain()
    {
        // =========================================================================
        // Simulation Parameters
        // =========================================================================

        // Physics time step (smaller = more accurate but slower)
        const double simulation_time_step = 2e-4; // [s]

        // Total simulation duration
        const double simulation_duration = 15.0; // [s]

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
        const double formation_radius = 0.5;  // [m] - horizontal distance from payload center
        const double attachment_radius = payload_radius * 0.7;  // Attachment points on payload top

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
            quad_configs[i].rope_length = 1.0;  // Can vary per quad if desired
        }

        // =========================================================================
        // Trajectory Waypoints (shared by all quadcopters, plus formation offset)
        // =========================================================================

        std::vector<TrajectoryWaypoint> waypoints;

        // Phase 1: Hover at initial position
        waypoints.push_back({Eigen::Vector3d(0, 0, initial_altitude), 0.0, 1.0});

        // Phase 2: Ascend to lift payload
        waypoints.push_back({Eigen::Vector3d(0, 0, 3.0), 4.0, 2.0});

        // Phase 3: Translate horizontally
        waypoints.push_back({Eigen::Vector3d(2.0, 1.0, 3.0), 8.0, 2.0});

        // Phase 4: Move to final position
        waypoints.push_back({Eigen::Vector3d(2.0, 1.0, 2.0), 12.0, 3.0});

        // =========================================================================
        // Derived Rope Parameters
        // =========================================================================

        // Calculate segment stiffness to achieve desired maximum stretch under load
        // Each rope carries payload_weight / num_quadcopters
        const double max_stretch_percentage = 0.15; // 15% stretch
        const double load_per_rope = (payload_mass * gravity) / num_quadcopters;
        const double avg_rope_length = quad_configs[0].rope_length;

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
        // Connect Systems
        // -------------------------------------------------------------------------

        // Force combiner: 2 inputs per quadcopter (rope + controller)
        auto& force_combiner = *builder.AddSystem<ExternallyAppliedSpatialForceMultiplexer>(
            2 * num_quadcopters);

        for (int q = 0; q < num_quadcopters; ++q) {
            // Rope system connections
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
        }

        // Apply combined forces to plant
        builder.Connect(force_combiner.get_output_port(),
                        plant.get_applied_spatial_force_input_port());

        // Log tension data (just first rope for now)
        auto& tension_logger = *builder.AddSystem<VectorLogSink<double>>(4);
        builder.Connect(rope_systems[0]->get_tension_output_port(),
                        tension_logger.get_input_port());

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

        // Publish initial state
        diagram->ForcedPublish(context);

        // -------------------------------------------------------------------------
        // Run Simulation
        // -------------------------------------------------------------------------

        simulator.set_target_realtime_rate(1.0);

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

        return 0;
    }

} // namespace quad_rope_lift

int main()
{
    return quad_rope_lift::DoMain();
}
