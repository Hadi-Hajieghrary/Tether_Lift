/// @file main.cc
/// Drake simulation: A quadcopter lifts a payload using a flexible rope/tether.
///
/// === Physical Model ===
/// - Quadcopter: A free-floating 6-DoF rigid body with thrust and torque control.
/// - Payload: A spherical rigid body resting on the ground with friction.
/// - Rope/Tether: Modeled as a bead-chain with (num_beads + 1) tension-only
///   spring-damper segments.
///
/// === Control Strategy ===
/// The quadcopter uses a cascaded controller:
/// 1. Altitude controller: PD control to follow a desired height trajectory.
/// 2. Attitude controller: PD control to keep the quadcopter level (upright).
/// 3. Tension-aware pickup: Smooth load transfer using rope tension feedback.
///
/// === Usage ===
///   ./build/quad_rope_lift
///
/// The simulation opens a Meshcat visualizer in your browser at http://localhost:7000

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>

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

    int DoMain()
    {
        // =========================================================================
        // Simulation Parameters
        // =========================================================================

        // Physics time step (smaller = more accurate but slower)
        const double simulation_time_step = 2e-4; // [s]

        // Total simulation duration
        const double simulation_duration = 8.0; // [s]

        // =========================================================================
        // Physical Properties
        // =========================================================================

        // Quadcopter
        const double quadcopter_mass = 1.5;                            // [kg]
        const Eigen::Vector3d quadcopter_dimensions(0.30, 0.30, 0.10); // [m]

        // Payload
        const double payload_mass = 2.0;    // [kg]
        const double payload_radius = 0.12; // [m]

        // Rope
        const double rope_rest_length = 1.0; // [m]
        const double rope_total_mass = 0.2;  // [kg]
        const int num_rope_beads = 10;

        // Trajectory
        const double initial_hover_altitude = 1.0; // [m]
        const double final_hover_altitude = 3.0;   // [m]

        // =========================================================================
        // Derived Rope Parameters
        // =========================================================================

        // Calculate segment stiffness to achieve desired maximum stretch under load
        const double max_stretch_percentage = 0.15; // 15% stretch
        const double gravity = 9.81;

        // Effective rope stiffness
        const double effective_rope_stiffness =
            (payload_mass * gravity) / (rope_rest_length * max_stretch_percentage);

        // Individual segment stiffness
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
            plant.AddModelInstance("quadcopter_payload_system");

        // -------------------------------------------------------------------------
        // Create the Quadcopter Body
        // -------------------------------------------------------------------------

        const SpatialInertia<double> quad_inertia(
            quadcopter_mass,
            Eigen::Vector3d::Zero(),
            UnitInertia<double>::SolidBox(
                quadcopter_dimensions[0],
                quadcopter_dimensions[1],
                quadcopter_dimensions[2]));

        const RigidBody<double> &quadcopter_body =
            plant.AddRigidBody("quadcopter", model_instance, quad_inertia);

        // Visuals: weld Drake's quadrotor URDF model onto our simplified dynamic body.
        // This keeps our dynamics but uses the quadrotor mesh for visuals.
        // Use the model bundled with Drake binary (no external download required).
        Parser parser(&plant);
        const std::vector<ModelInstanceIndex> visual_instances =
            parser.AddModels(
                drake::FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"));

        const RigidBody<double> &visual_base_link =
            plant.GetBodyByName("base_link", visual_instances[0]);
        plant.WeldFrames(
            quadcopter_body.body_frame(),
            visual_base_link.body_frame(),
            RigidTransformd::Identity());

        // -------------------------------------------------------------------------
        // Create the Payload Body
        // -------------------------------------------------------------------------

        const SpatialInertia<double> payload_inertia(
            payload_mass,
            Eigen::Vector3d::Zero(),
            UnitInertia<double>::SolidSphere(payload_radius));

        const RigidBody<double> &payload_body =
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
        // Create the Rope Beads
        // -------------------------------------------------------------------------

        const RopeParameters rope_params = ComputeRopeParameters(
            num_rope_beads,
            rope_rest_length,
            rope_total_mass,
            segment_stiffness,
            segment_damping,
            true,   // bead_diameter_equals_spacing
            0.015); // max_bead_radius

        const SpatialInertia<double> bead_inertia(
            rope_params.bead_mass,
            Eigen::Vector3d::Zero(),
            UnitInertia<double>::SolidSphere(rope_params.bead_radius));

        std::vector<const RigidBody<double> *> bead_bodies;
        bead_bodies.reserve(num_rope_beads);

        for (int i = 0; i < num_rope_beads; ++i)
        {
            const std::string name = "rope_bead_" + std::to_string(i);

            const RigidBody<double> &bead =
                plant.AddRigidBody(name, model_instance, bead_inertia);

            plant.RegisterCollisionGeometry(
                bead,
                RigidTransformd::Identity(),
                Sphere(rope_params.bead_radius),
                name + "_collision",
                ground_friction);

            plant.RegisterVisualGeometry(
                bead,
                RigidTransformd::Identity(),
                Sphere(rope_params.bead_radius),
                name + "_visual",
                Eigen::Vector4d(0.25, 0.25, 0.25, 1.0)); // Dark gray

            bead_bodies.push_back(&bead);
        }

        // Finalize the plant
        plant.Finalize();

        // -------------------------------------------------------------------------
        // Define Rope Attachment Points
        // -------------------------------------------------------------------------

        const Eigen::Vector3d quad_attachment_offset(
            0.0, 0.0, -quadcopter_dimensions[2] / 2.0 - 0.05);

        const Eigen::Vector3d payload_attachment_offset(0.0, 0.0, payload_radius);

        // -------------------------------------------------------------------------
        // Create Rope Force System
        // -------------------------------------------------------------------------

        auto &rope_system = *builder.AddSystem<RopeForceSystem>(
            plant,
            quadcopter_body,
            payload_body,
            bead_bodies,
            quad_attachment_offset,
            payload_attachment_offset,
            rope_rest_length,
            rope_params.segment_stiffness,
            rope_params.segment_damping);

        // Zero-order hold for tension signal
        auto &tension_hold = *builder.AddSystem<ZeroOrderHold<double>>(
            simulation_time_step, 4);
        builder.Connect(rope_system.get_tension_output_port(),
                        tension_hold.get_input_port());

        // -------------------------------------------------------------------------
        // Create Quadcopter Controller
        // -------------------------------------------------------------------------

        ControllerParams ctrl_params;
        ctrl_params.initial_altitude = initial_hover_altitude;
        ctrl_params.final_altitude = final_hover_altitude;
        ctrl_params.pickup_target_tension = payload_mass * gravity;

        auto &controller = *builder.AddSystem<QuadcopterLiftController>(
            plant, quadcopter_body, ctrl_params);

        // -------------------------------------------------------------------------
        // Connect Systems
        // -------------------------------------------------------------------------

        // Force combiner
        auto &force_combiner = *builder.AddSystem<ExternallyAppliedSpatialForceMultiplexer>(2);

        // Rope system connections
        builder.Connect(plant.get_state_output_port(),
                        rope_system.get_plant_state_input_port());
        builder.Connect(rope_system.get_forces_output_port(),
                        force_combiner.get_input_port(1));

        // Controller connections
        builder.Connect(plant.get_state_output_port(),
                        controller.get_plant_state_input_port());
        builder.Connect(tension_hold.get_output_port(),
                        controller.get_tension_input_port());
        builder.Connect(controller.get_control_output_port(),
                        force_combiner.get_input_port(0));

        // Apply combined forces to plant
        builder.Connect(force_combiner.get_output_port(),
                        plant.get_applied_spatial_force_input_port());

        // Log tension data
        auto &tension_logger = *builder.AddSystem<VectorLogSink<double>>(4);
        builder.Connect(rope_system.get_tension_output_port(),
                        tension_logger.get_input_port());

        // -------------------------------------------------------------------------
        // Set Up Visualization
        // -------------------------------------------------------------------------

        auto meshcat = std::make_shared<Meshcat>();
        meshcat->Delete();
        MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat);

        // Add rope polyline visualizer
        std::vector<std::pair<const RigidBody<double> *, Eigen::Vector3d>> rope_path_points;
        rope_path_points.reserve(num_rope_beads + 2);
        rope_path_points.emplace_back(&quadcopter_body, quad_attachment_offset);
        for (const auto *bead : bead_bodies)
        {
            rope_path_points.emplace_back(bead, Eigen::Vector3d::Zero());
        }
        rope_path_points.emplace_back(&payload_body, payload_attachment_offset);

        auto &rope_visualizer = *builder.AddSystem<RopeVisualizer>(
            plant, rope_path_points, meshcat, "rope_line");
        builder.Connect(plant.get_state_output_port(),
                        rope_visualizer.get_plant_state_input_port());

        // -------------------------------------------------------------------------
        // Build and Initialize Simulation
        // -------------------------------------------------------------------------

        auto diagram = builder.Build();
        Simulator<double> simulator(*diagram);
        auto &context = simulator.get_mutable_context();
        auto &plant_context = plant.GetMyMutableContextFromRoot(&context);

        // Update controller mass to include visual model mass (same as Python version)
        const double visual_model_mass =
            plant.CalcTotalMass(plant_context, visual_instances);
        controller.set_mass(quadcopter_mass + visual_model_mass);

        // Set initial poses
        const double ground_clearance = 0.01;

        plant.SetFreeBodyPose(
            &plant_context,
            payload_body,
            RigidTransformd(Eigen::Vector3d(0.0, 0.0, payload_radius + ground_clearance)));

        plant.SetFreeBodyPose(
            &plant_context,
            quadcopter_body,
            RigidTransformd(Eigen::Vector3d(0.0, 0.0, initial_hover_altitude)));

        // Initialize rope beads in slack configuration
        const auto &quad_pose = plant.EvalBodyPoseInWorld(plant_context, quadcopter_body);
        const auto &payload_pose = plant.EvalBodyPoseInWorld(plant_context, payload_body);
        const Eigen::Vector3d rope_start_world = quad_pose * quad_attachment_offset;
        const Eigen::Vector3d rope_end_world = payload_pose * payload_attachment_offset;

        const auto initial_bead_positions = GenerateSlackRopePositions(
            rope_start_world,
            rope_end_world,
            num_rope_beads,
            rope_rest_length,
            0.85,                     // slack_ratio
            Eigen::Vector3d::UnitX(), // lateral_direction
            1.5);                     // max_lateral_amplitude

        for (int i = 0; i < num_rope_beads; ++i)
        {
            plant.SetFreeBodyPose(
                &plant_context,
                *bead_bodies[i],
                RigidTransformd(initial_bead_positions[i]));
        }

        // Publish initial state
        diagram->ForcedPublish(context);

        // -------------------------------------------------------------------------
        // Run Simulation
        // -------------------------------------------------------------------------

        simulator.set_target_realtime_rate(1.0);

        std::cout << "Starting simulation for " << simulation_duration << "s...\n";
        std::cout << "Open Meshcat at: http://localhost:7000\n";

        // Run in chunks to show progress
        const double progress_interval = 0.1;
        double current_time = 0.0;

        while (current_time < simulation_duration)
        {
            current_time += progress_interval;
            simulator.AdvanceTo(current_time);

            // Print progress every second
            if (static_cast<int>(current_time * 10) % 10 == 0)
            {
                std::cout << "  Simulated " << current_time << "s / "
                          << simulation_duration << "s\n";
            }
        }

        // -------------------------------------------------------------------------
        // Print Results
        // -------------------------------------------------------------------------

        const auto &tension_log = tension_logger.FindLog(context);
        const auto &tension_data = tension_log.data();

        double max_tension = 0.0;
        for (int i = 0; i < tension_data.cols(); ++i)
        {
            max_tension = std::max(max_tension, tension_data(0, i));
        }

        std::cout << "\nSimulation complete!\n";
        std::cout << "  Max rope tension: " << max_tension << " N\n";
        std::cout << "  Payload weight: " << payload_mass * gravity << " N\n";

        return 0;
    }

} // namespace quad_rope_lift

int main()
{
    return quad_rope_lift::DoMain();
}
