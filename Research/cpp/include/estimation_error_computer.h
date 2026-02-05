#pragma once

#include <Eigen/Core>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// System that computes estimation error by comparing estimated state to ground truth.
///
/// Input ports:
///   - plant_state: Full state vector from MultibodyPlant (ground truth)
///   - estimated_state: Estimated [px, py, pz, vx, vy, vz]
///
/// Output port:
///   - error: [pos_err_x, pos_err_y, pos_err_z, vel_err_x, vel_err_y, vel_err_z,
///             pos_norm_err, vel_norm_err]
///
class EstimationErrorComputer final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EstimationErrorComputer);

  /// Constructs the error computer for a specific body.
  EstimationErrorComputer(const drake::multibody::MultibodyPlant<double>& plant,
                          const drake::multibody::RigidBody<double>& body);

  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  const drake::systems::InputPort<double>& get_estimated_state_input_port() const {
    return get_input_port(estimated_state_port_);
  }

  const drake::systems::OutputPort<double>& get_error_output_port() const {
    return get_output_port(error_port_);
  }

 private:
  void CalcError(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* output) const;

  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::multibody::BodyIndex body_index_;

  int plant_state_port_{-1};
  int estimated_state_port_{-1};
  int error_port_{-1};
};

}  // namespace quad_rope_lift
