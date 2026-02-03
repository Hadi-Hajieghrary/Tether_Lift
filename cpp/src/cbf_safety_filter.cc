#include "cbf_safety_filter.h"

#include <algorithm>
#include <cmath>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;

CbfSafetyFilter::CbfSafetyFilter(double quadcopter_mass,
                                 int max_neighbors,
                                 const CbfSafetyParams& params)
    : params_(params),
      quadcopter_mass_(quadcopter_mass),
      max_neighbors_(max_neighbors) {

  // Input ports
  nominal_thrust_port_ =
      DeclareVectorInputPort("nominal_thrust", BasicVector<double>(1))
          .get_index();

  nominal_torque_port_ =
      DeclareVectorInputPort("nominal_torque", BasicVector<double>(3))
          .get_index();

  // Quad state: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz] = 13
  quad_state_port_ =
      DeclareVectorInputPort("quad_state", BasicVector<double>(13))
          .get_index();

  cable_tension_port_ =
      DeclareVectorInputPort("cable_tension", BasicVector<double>(1))
          .get_index();

  cable_direction_port_ =
      DeclareVectorInputPort("cable_direction", BasicVector<double>(3))
          .get_index();

  // Neighbor positions: 3 * max_neighbors
  neighbor_positions_port_ =
      DeclareVectorInputPort("neighbor_positions",
                             BasicVector<double>(3 * max_neighbors))
          .get_index();

  // Output ports
  filtered_thrust_port_ =
      DeclareVectorOutputPort("filtered_thrust", BasicVector<double>(1),
                              &CbfSafetyFilter::CalcFilteredThrust)
          .get_index();

  filtered_torque_port_ =
      DeclareVectorOutputPort("filtered_torque", BasicVector<double>(3),
                              &CbfSafetyFilter::CalcFilteredTorque)
          .get_index();

  constraint_active_port_ =
      DeclareVectorOutputPort("constraint_active", BasicVector<double>(kNumBarriers),
                              &CbfSafetyFilter::CalcConstraintActive)
          .get_index();

  barrier_values_port_ =
      DeclareVectorOutputPort("barrier_values", BasicVector<double>(kNumBarriers),
                              &CbfSafetyFilter::CalcBarrierValues)
          .get_index();
}

double CbfSafetyFilter::ComputeTensionLowerBarrier(double T) const {
  // h_Tmin = T - T_min
  // Safe when h > 0 (tension above minimum)
  return T - params_.min_tension;
}

double CbfSafetyFilter::ComputeTensionUpperBarrier(double T) const {
  // h_Tmax = T_max - T
  // Safe when h > 0 (tension below maximum)
  return params_.max_tension - T;
}

double CbfSafetyFilter::ComputeCableAngleBarrier(const Eigen::Vector3d& cable_dir) const {
  // Cable direction points from quad toward load
  // cos(theta) = -cable_dir.z() (angle from vertical)
  // h_angle = cos(theta) - cos(theta_max)
  // Safe when h > 0 (angle within limit)
  double cos_theta = -cable_dir.z();  // Positive when cable points down
  double cos_max = std::cos(params_.max_cable_angle);
  return cos_theta - cos_max;
}

double CbfSafetyFilter::ComputeCollisionBarrier(
    const Eigen::Vector3d& my_pos,
    const Eigen::Vector3d& neighbor_pos) const {
  // h_coll = ||r_i - r_j||² - d_min²
  // Safe when h > 0 (distance above minimum)
  double dist_sq = (my_pos - neighbor_pos).squaredNorm();
  double min_dist_sq = params_.min_quad_spacing * params_.min_quad_spacing;
  return dist_sq - min_dist_sq;
}

CbfSafetyFilter::QpSolution CbfSafetyFilter::SolveQp(
    double nominal_thrust,
    const Eigen::Vector3d& nominal_torque,
    const Eigen::Vector3d& quad_pos,
    const Eigen::Vector3d& quad_vel,
    double cable_tension,
    const Eigen::Vector3d& cable_dir,
    const std::vector<Eigen::Vector3d>& neighbor_positions) const {

  QpSolution sol;
  sol.filtered_thrust = nominal_thrust;
  sol.filtered_torque = nominal_torque;
  sol.constraint_active.setZero();
  sol.success = true;

  // Compute barrier values
  double h_Tmin = ComputeTensionLowerBarrier(cable_tension);
  double h_Tmax = ComputeTensionUpperBarrier(cable_tension);
  double h_angle = ComputeCableAngleBarrier(cable_dir);

  // Find minimum collision barrier
  double h_coll = 1e6;  // Large positive = safe
  for (const auto& neighbor_pos : neighbor_positions) {
    double h = ComputeCollisionBarrier(quad_pos, neighbor_pos);
    h_coll = std::min(h_coll, h);
  }

  // === Simplified CBF filtering (analytical solution for 1D control) ===
  // For thrust: we only modify thrust to satisfy tension constraints
  // Full QP would consider coupled dynamics, but this is a practical approximation

  double thrust = nominal_thrust;

  // --- Tension lower bound constraint ---
  // If tension is too low, need more thrust to increase tension
  // dh_Tmin/dt >= -alpha * h_Tmin
  // Approximately: dT/dt ≈ (thrust_change) * sensitivity
  // For cable dynamics: T ≈ m_load * (g + a_vertical) / N
  // Sensitivity: dT/d(thrust) ≈ 1 (rough approximation)
  if (h_Tmin < params_.tension_activation_margin) {
    double required_rate = -params_.cbf_alpha * h_Tmin;
    if (required_rate > 0) {
      // Need positive tension change - increase thrust
      double thrust_correction = required_rate * quadcopter_mass_;
      thrust = std::max(thrust, nominal_thrust + thrust_correction);
      sol.constraint_active[0] = 1.0;
    }
  }

  // --- Tension upper bound constraint ---
  // If tension is too high, reduce thrust to decrease tension
  if (h_Tmax < params_.tension_activation_margin) {
    double required_rate = -params_.cbf_alpha * h_Tmax;
    if (required_rate > 0) {
      // Need negative tension change - decrease thrust
      double thrust_correction = required_rate * quadcopter_mass_;
      thrust = std::min(thrust, nominal_thrust - thrust_correction);
      sol.constraint_active[1] = 1.0;
    }
  }

  // --- Cable angle constraint ---
  // If cable angle is too large, adjust horizontal forces via torque
  if (h_angle < params_.angle_activation_margin) {
    double required_rate = -params_.cbf_alpha * h_angle;
    if (required_rate > 0) {
      // Need to reduce horizontal force - reduce tilt
      // Scale down torques that would increase tilt
      double scale = std::max(0.5, 1.0 - required_rate);
      sol.filtered_torque = nominal_torque * scale;
      sol.constraint_active[2] = 1.0;
    }
  }

  // --- Collision avoidance constraint ---
  if (h_coll < params_.collision_barrier_margin * params_.collision_barrier_margin) {
    // Near collision - this should activate position control layer
    // For now, reduce aggressive maneuvers
    sol.filtered_torque *= 0.5;
    sol.constraint_active[3] = 1.0;
  }

  // Apply actuator limits
  sol.filtered_thrust = std::clamp(thrust, min_thrust_, max_thrust_);
  for (int i = 0; i < 3; ++i) {
    sol.filtered_torque[i] = std::clamp(sol.filtered_torque[i],
                                        -max_torque_, max_torque_);
  }

  return sol;
}

void CbfSafetyFilter::CalcFilteredThrust(
    const Context<double>& context,
    BasicVector<double>* output) const {

  // Get inputs
  double nominal_thrust = get_input_port(nominal_thrust_port_).Eval(context)[0];
  const auto& torque_raw = get_input_port(nominal_torque_port_).Eval(context);
  Eigen::Vector3d nominal_torque(torque_raw[0], torque_raw[1], torque_raw[2]);

  const auto& state_raw = get_input_port(quad_state_port_).Eval(context);
  Eigen::Vector3d quad_pos(state_raw[0], state_raw[1], state_raw[2]);
  Eigen::Vector3d quad_vel(state_raw[7], state_raw[8], state_raw[9]);

  double cable_tension = get_input_port(cable_tension_port_).Eval(context)[0];
  const auto& dir_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d cable_dir(dir_raw[0], dir_raw[1], dir_raw[2]);

  // Get neighbor positions
  const auto& neighbors_raw = get_input_port(neighbor_positions_port_).Eval(context);
  std::vector<Eigen::Vector3d> neighbors;
  for (int i = 0; i < max_neighbors_; ++i) {
    Eigen::Vector3d pos(neighbors_raw[3*i], neighbors_raw[3*i+1], neighbors_raw[3*i+2]);
    if (pos.norm() > 0.1) {  // Non-zero position indicates valid neighbor
      neighbors.push_back(pos);
    }
  }

  // Solve QP
  QpSolution sol = SolveQp(nominal_thrust, nominal_torque, quad_pos, quad_vel,
                           cable_tension, cable_dir, neighbors);

  output->SetAtIndex(0, sol.filtered_thrust);
}

void CbfSafetyFilter::CalcFilteredTorque(
    const Context<double>& context,
    BasicVector<double>* output) const {

  // Get inputs (duplicate work with CalcFilteredThrust - could cache)
  double nominal_thrust = get_input_port(nominal_thrust_port_).Eval(context)[0];
  const auto& torque_raw = get_input_port(nominal_torque_port_).Eval(context);
  Eigen::Vector3d nominal_torque(torque_raw[0], torque_raw[1], torque_raw[2]);

  const auto& state_raw = get_input_port(quad_state_port_).Eval(context);
  Eigen::Vector3d quad_pos(state_raw[0], state_raw[1], state_raw[2]);
  Eigen::Vector3d quad_vel(state_raw[7], state_raw[8], state_raw[9]);

  double cable_tension = get_input_port(cable_tension_port_).Eval(context)[0];
  const auto& dir_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d cable_dir(dir_raw[0], dir_raw[1], dir_raw[2]);

  const auto& neighbors_raw = get_input_port(neighbor_positions_port_).Eval(context);
  std::vector<Eigen::Vector3d> neighbors;
  for (int i = 0; i < max_neighbors_; ++i) {
    Eigen::Vector3d pos(neighbors_raw[3*i], neighbors_raw[3*i+1], neighbors_raw[3*i+2]);
    if (pos.norm() > 0.1) {
      neighbors.push_back(pos);
    }
  }

  QpSolution sol = SolveQp(nominal_thrust, nominal_torque, quad_pos, quad_vel,
                           cable_tension, cable_dir, neighbors);

  output->SetFromVector(sol.filtered_torque);
}

void CbfSafetyFilter::CalcConstraintActive(
    const Context<double>& context,
    BasicVector<double>* output) const {

  // Same computation as above
  double nominal_thrust = get_input_port(nominal_thrust_port_).Eval(context)[0];
  const auto& torque_raw = get_input_port(nominal_torque_port_).Eval(context);
  Eigen::Vector3d nominal_torque(torque_raw[0], torque_raw[1], torque_raw[2]);

  const auto& state_raw = get_input_port(quad_state_port_).Eval(context);
  Eigen::Vector3d quad_pos(state_raw[0], state_raw[1], state_raw[2]);
  Eigen::Vector3d quad_vel(state_raw[7], state_raw[8], state_raw[9]);

  double cable_tension = get_input_port(cable_tension_port_).Eval(context)[0];
  const auto& dir_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d cable_dir(dir_raw[0], dir_raw[1], dir_raw[2]);

  const auto& neighbors_raw = get_input_port(neighbor_positions_port_).Eval(context);
  std::vector<Eigen::Vector3d> neighbors;
  for (int i = 0; i < max_neighbors_; ++i) {
    Eigen::Vector3d pos(neighbors_raw[3*i], neighbors_raw[3*i+1], neighbors_raw[3*i+2]);
    if (pos.norm() > 0.1) {
      neighbors.push_back(pos);
    }
  }

  QpSolution sol = SolveQp(nominal_thrust, nominal_torque, quad_pos, quad_vel,
                           cable_tension, cable_dir, neighbors);

  output->SetFromVector(sol.constraint_active);
}

void CbfSafetyFilter::CalcBarrierValues(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state_raw = get_input_port(quad_state_port_).Eval(context);
  Eigen::Vector3d quad_pos(state_raw[0], state_raw[1], state_raw[2]);

  double cable_tension = get_input_port(cable_tension_port_).Eval(context)[0];
  const auto& dir_raw = get_input_port(cable_direction_port_).Eval(context);
  Eigen::Vector3d cable_dir(dir_raw[0], dir_raw[1], dir_raw[2]);

  const auto& neighbors_raw = get_input_port(neighbor_positions_port_).Eval(context);

  // Compute barrier values
  double h_Tmin = ComputeTensionLowerBarrier(cable_tension);
  double h_Tmax = ComputeTensionUpperBarrier(cable_tension);
  double h_angle = ComputeCableAngleBarrier(cable_dir);

  // Find minimum collision barrier
  double h_coll = 1e6;
  for (int i = 0; i < max_neighbors_; ++i) {
    Eigen::Vector3d pos(neighbors_raw[3*i], neighbors_raw[3*i+1], neighbors_raw[3*i+2]);
    if (pos.norm() > 0.1) {
      double h = ComputeCollisionBarrier(quad_pos, pos);
      h_coll = std::min(h_coll, h);
    }
  }

  Eigen::Vector4d barriers(h_Tmin, h_Tmax, h_angle, h_coll);
  output->SetFromVector(barriers);
}

}  // namespace quad_rope_lift
