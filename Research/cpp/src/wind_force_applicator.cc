/// @file wind_force_applicator.cc
/// @brief Converts wind velocities into aerodynamic drag forces on drone and payload bodies.

#include "wind_force_applicator.h"

#include <cmath>

namespace quad_rope_lift {

using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::SpatialForce;
using drake::systems::Context;

WindForceApplicator::WindForceApplicator(
    const std::vector<const drake::multibody::RigidBody<double>*>& drone_bodies,
    const drake::multibody::RigidBody<double>& payload_body,
    int num_drones,
    const DragParams& params)
    : payload_body_index_(payload_body.index()),
      params_(params),
      num_drones_(num_drones) {

  for (const auto* body : drone_bodies) {
    drone_body_indices_.push_back(body->index());
  }

  wind_port_ = DeclareVectorInputPort(
      "wind_velocities", 3 * num_drones_).get_index();

  forces_port_ = DeclareAbstractOutputPort(
      "spatial_forces",
      []() {
        return drake::AbstractValue::Make(
            std::vector<ExternallyAppliedSpatialForce<double>>());
      },
      [this](const Context<double>& context,
             drake::AbstractValue* output) {
        auto& forces = output->get_mutable_value<
            std::vector<ExternallyAppliedSpatialForce<double>>>();
        this->CalcSpatialForces(context, &forces);
      }).get_index();
}

void WindForceApplicator::CalcSpatialForces(
    const Context<double>& context,
    std::vector<ExternallyAppliedSpatialForce<double>>* forces) const {

  forces->clear();

  const Eigen::VectorXd& wind = get_input_port(wind_port_).Eval(context);

  // Apply drag to each drone: F = 0.5 * rho * Cd * A * |v_wind|^2 * v_hat
  for (int q = 0; q < num_drones_; ++q) {
    const Eigen::Vector3d wind_vel = wind.segment<3>(3 * q);
    const double v_mag = wind_vel.norm();

    Eigen::Vector3d F_drag = Eigen::Vector3d::Zero();
    if (v_mag > 0.01) {
      const double F_mag = 0.5 * params_.rho * params_.Cd_drone *
                           params_.A_drone * v_mag * v_mag;
      F_drag = F_mag * (wind_vel / v_mag);
    }

    ExternallyAppliedSpatialForce<double> sf;
    sf.body_index = drone_body_indices_[q];
    sf.p_BoBq_B = Eigen::Vector3d::Zero();
    sf.F_Bq_W = SpatialForce<double>(Eigen::Vector3d::Zero(), F_drag);
    forces->push_back(sf);
  }

  // Apply drag to payload using average wind
  Eigen::Vector3d avg_wind = Eigen::Vector3d::Zero();
  for (int q = 0; q < num_drones_; ++q) {
    avg_wind += wind.segment<3>(3 * q);
  }
  avg_wind /= num_drones_;

  const double v_mag_p = avg_wind.norm();
  Eigen::Vector3d F_payload = Eigen::Vector3d::Zero();
  if (v_mag_p > 0.01) {
    const double F_mag = 0.5 * params_.rho * params_.Cd_payload *
                         params_.A_payload * v_mag_p * v_mag_p;
    F_payload = F_mag * (avg_wind / v_mag_p);
  }

  ExternallyAppliedSpatialForce<double> sf_p;
  sf_p.body_index = payload_body_index_;
  sf_p.p_BoBq_B = Eigen::Vector3d::Zero();
  sf_p.F_Bq_W = SpatialForce<double>(Eigen::Vector3d::Zero(), F_payload);
  forces->push_back(sf_p);
}

}  // namespace quad_rope_lift
