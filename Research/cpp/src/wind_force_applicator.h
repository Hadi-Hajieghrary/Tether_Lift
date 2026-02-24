#pragma once

/// @file wind_force_applicator.h
/// @brief Converts wind velocities to aerodynamic drag forces on drone and payload bodies.
///
/// Drag model: F_drag = 0.5 * rho * Cd * A * |v_wind|^2 * v_wind_hat
/// Applied as ExternallyAppliedSpatialForce to the Drake plant.

#include <Eigen/Core>
#include <vector>

#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Parameters for aerodynamic drag.
struct DragParams {
  double rho = 1.225;       ///< Air density [kg/m^3]
  double Cd_drone = 1.0;    ///< Drag coefficient for quadrotor
  double A_drone = 0.04;    ///< Frontal area of quadrotor [m^2]
  double Cd_payload = 0.47; ///< Drag coefficient for spherical payload
  double A_payload = 0.07;  ///< Frontal area of payload [m^2] (pi*r^2)
};

/// Converts wind velocities to spatial forces on drone and payload bodies.
///
/// Input ports:
///   - wind_velocities [3*N]: Wind velocity at each drone position
///
/// Output ports:
///   - spatial_forces: vector<ExternallyAppliedSpatialForce> (abstract)
class WindForceApplicator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WindForceApplicator);

  WindForceApplicator(
      const std::vector<const drake::multibody::RigidBody<double>*>& drone_bodies,
      const drake::multibody::RigidBody<double>& payload_body,
      int num_drones,
      const DragParams& params = DragParams());

  const drake::systems::InputPort<double>& get_wind_input_port() const {
    return get_input_port(wind_port_);
  }

  const drake::systems::OutputPort<double>& get_forces_output_port() const {
    return get_output_port(forces_port_);
  }

 private:
  void CalcSpatialForces(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>* output) const;

  std::vector<drake::multibody::BodyIndex> drone_body_indices_;
  drake::multibody::BodyIndex payload_body_index_;
  DragParams params_;
  int num_drones_;

  int wind_port_{};
  int forces_port_{};
};

}  // namespace quad_rope_lift
