#include "wind_disturbance.h"

#include <algorithm>
#include <cmath>

namespace quad_rope_lift {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;

// === WindDisturbance implementation ===

WindDisturbance::WindDisturbance(int num_drones,
                                 const DrydenTurbulenceParams& turbulence_params,
                                 const GustParams& gust_params,
                                 double dt)
    : turbulence_params_(turbulence_params),
      gust_params_(gust_params),
      num_drones_(num_drones),
      dt_(dt),
      rng_(42),  // Default seed
      normal_dist_(0.0, 1.0),
      gust_interval_dist_(1.0 / gust_params.mean_interval) {

  // Input port: drone positions [3*N]
  positions_port_ =
      DeclareVectorInputPort("drone_positions",
                             BasicVector<double>(3 * num_drones))
          .get_index();

  // Initialize discrete state
  // [turbulence states (3 per drone), gust state (6)]
  int state_size = kTurbStatePerDrone * num_drones + kGustStateSize;
  Eigen::VectorXd initial_state(state_size);
  initial_state.setZero();

  state_index_ = DeclareDiscreteState(initial_state);

  // Periodic update for turbulence
  DeclarePeriodicDiscreteUpdateEvent(dt_, 0.0,
                                     &WindDisturbance::UpdateTurbulence);

  // Output port: wind velocities [3*N]
  velocities_port_ =
      DeclareVectorOutputPort("wind_velocities",
                              BasicVector<double>(3 * num_drones),
                              &WindDisturbance::CalcWindVelocities)
          .get_index();
}

void WindDisturbance::set_seed(unsigned int seed) {
  rng_.seed(seed);
}

double WindDisturbance::ComputeAltitudeScale(double altitude) const {
  if (!turbulence_params_.altitude_dependent) {
    return 1.0;
  }

  // Low altitude model: intensity increases with height up to scale height
  // σ(h) = σ_ref * (h / h_ref)^(1/6) for h < 1000 ft
  double h = std::max(0.1, altitude);
  double h_ref = turbulence_params_.altitude_scale_height;

  if (h < h_ref) {
    return std::pow(h / h_ref, 1.0 / 6.0);
  }
  return 1.0;
}

double WindDisturbance::ComputeSpatialCorrelation(
    const Eigen::Vector3d& pos1,
    const Eigen::Vector3d& pos2) const {

  // Exponential decay with distance
  double dist = (pos1 - pos2).norm();
  return std::exp(-dist / spatial_correlation_scale_);
}

void WindDisturbance::GenerateTurbulenceComponent(
    double sigma, double L, double V,
    double dt, double& state, double white_noise) const {

  // First-order Dryden turbulence filter
  // Transfer function: H(s) = σ * sqrt(2V/L) / (s + V/L)
  // Discrete approximation: x[k+1] = α*x[k] + β*w[k]

  double tau = L / V;  // Time constant
  double alpha = std::exp(-dt / tau);
  double beta = sigma * std::sqrt(1.0 - alpha * alpha);

  state = alpha * state + beta * white_noise;
}

Eigen::Vector3d WindDisturbance::ComputeGust(double t) const {
  // Placeholder - gust computation based on state
  // Would read from discrete state for active gust
  return Eigen::Vector3d::Zero();
}

EventStatus WindDisturbance::UpdateTurbulence(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {

  auto& state = discrete_state->get_mutable_vector(state_index_);

  // Get drone positions
  const auto& positions_raw = get_input_port(positions_port_).Eval(context);

  // Generate correlated white noise for each drone
  std::vector<Eigen::Vector3d> white_noises(num_drones_);
  for (int i = 0; i < num_drones_; ++i) {
    white_noises[i] = Eigen::Vector3d(
        normal_dist_(rng_), normal_dist_(rng_), normal_dist_(rng_));
  }

  // Apply spatial correlation (simplified: average with neighbors)
  for (int i = 0; i < num_drones_; ++i) {
    Eigen::Vector3d pos_i(positions_raw[3*i], positions_raw[3*i+1], positions_raw[3*i+2]);

    Eigen::Vector3d correlated_noise = white_noises[i];
    double total_weight = 1.0;

    for (int j = 0; j < num_drones_; ++j) {
      if (i != j) {
        Eigen::Vector3d pos_j(positions_raw[3*j], positions_raw[3*j+1], positions_raw[3*j+2]);
        double corr = ComputeSpatialCorrelation(pos_i, pos_j);
        correlated_noise += corr * white_noises[j];
        total_weight += corr;
      }
    }
    correlated_noise /= total_weight;

    // Update turbulence states for drone i
    int base_idx = kTurbStatePerDrone * i;
    double u_state = state[base_idx];
    double v_state = state[base_idx + 1];
    double w_state = state[base_idx + 2];

    double altitude = pos_i.z();
    double alt_scale = ComputeAltitudeScale(altitude);

    GenerateTurbulenceComponent(
        turbulence_params_.sigma_u * alt_scale,
        turbulence_params_.Lu, reference_airspeed_,
        dt_, u_state, correlated_noise.x());

    GenerateTurbulenceComponent(
        turbulence_params_.sigma_v * alt_scale,
        turbulence_params_.Lv, reference_airspeed_,
        dt_, v_state, correlated_noise.y());

    GenerateTurbulenceComponent(
        turbulence_params_.sigma_w * alt_scale,
        turbulence_params_.Lw, reference_airspeed_,
        dt_, w_state, correlated_noise.z());

    state[base_idx] = u_state;
    state[base_idx + 1] = v_state;
    state[base_idx + 2] = w_state;
  }

  // Update gust state (if enabled)
  if (gust_params_.enabled) {
    int gust_base = kTurbStatePerDrone * num_drones_;
    double gust_active = state[gust_base];
    double gust_start = state[gust_base + 1];

    if (gust_active < 0.5) {
      // Check for new gust (random trigger)
      double p_gust = 1.0 - std::exp(-dt_ / gust_params_.mean_interval);
      if (normal_dist_(rng_) < p_gust * 3.0) {  // Approximate
        // Start new gust
        state[gust_base] = 1.0;
        state[gust_base + 1] = context.get_time();

        // Random direction (mostly horizontal)
        double theta = 2.0 * M_PI * (normal_dist_(rng_) + 1.0) / 2.0;
        state[gust_base + 2] = std::cos(theta);
        state[gust_base + 3] = std::sin(theta);
        state[gust_base + 4] = 0.1 * normal_dist_(rng_);

        // Random magnitude
        state[gust_base + 5] = gust_params_.max_magnitude *
                               (0.5 + 0.5 * std::abs(normal_dist_(rng_)));
      }
    } else {
      // Check if gust has ended
      double elapsed = context.get_time() - gust_start;
      double total_gust_time = gust_params_.rise_time +
                               gust_params_.hold_time +
                               gust_params_.fall_time;
      if (elapsed > total_gust_time) {
        state[gust_base] = 0.0;
      }
    }
  }

  return EventStatus::Succeeded();
}

void WindDisturbance::CalcWindVelocities(
    const Context<double>& context,
    BasicVector<double>* output) const {

  const auto& state = context.get_discrete_state(state_index_).value();
  const auto& positions_raw = get_input_port(positions_port_).Eval(context);

  Eigen::VectorXd wind_out(3 * num_drones_);

  for (int i = 0; i < num_drones_; ++i) {
    int base_idx = kTurbStatePerDrone * i;

    // Turbulence components
    Eigen::Vector3d turbulence(state[base_idx],
                               state[base_idx + 1],
                               state[base_idx + 2]);

    // Mean wind
    Eigen::Vector3d wind = turbulence_params_.mean_wind + turbulence;

    // Add gust if active
    if (gust_params_.enabled) {
      int gust_base = kTurbStatePerDrone * num_drones_;
      double gust_active = state[gust_base];

      if (gust_active > 0.5) {
        double gust_start = state[gust_base + 1];
        Eigen::Vector3d gust_dir(state[gust_base + 2],
                                 state[gust_base + 3],
                                 state[gust_base + 4]);
        gust_dir.normalize();
        double gust_mag = state[gust_base + 5];

        double t_elapsed = context.get_time() - gust_start;

        // Gust envelope: rise - hold - fall
        double envelope = 0.0;
        if (t_elapsed < gust_params_.rise_time) {
          envelope = t_elapsed / gust_params_.rise_time;
        } else if (t_elapsed < gust_params_.rise_time + gust_params_.hold_time) {
          envelope = 1.0;
        } else {
          double t_fall = t_elapsed - gust_params_.rise_time - gust_params_.hold_time;
          envelope = std::max(0.0, 1.0 - t_fall / gust_params_.fall_time);
        }

        wind += envelope * gust_mag * gust_dir;
      }
    }

    wind_out.segment<3>(3 * i) = wind;
  }

  output->SetFromVector(wind_out);
}

// === ConstantWind implementation ===

ConstantWind::ConstantWind(const Eigen::Vector3d& wind_velocity, int num_drones)
    : wind_velocity_(wind_velocity), num_drones_(num_drones) {

  wind_port_ =
      DeclareVectorOutputPort("wind_velocities",
                              BasicVector<double>(3 * num_drones),
                              &ConstantWind::CalcWind)
          .get_index();
}

void ConstantWind::CalcWind(const Context<double>& context,
                            BasicVector<double>* output) const {
  Eigen::VectorXd wind_out(3 * num_drones_);
  for (int i = 0; i < num_drones_; ++i) {
    wind_out.segment<3>(3 * i) = wind_velocity_;
  }
  output->SetFromVector(wind_out);
}

}  // namespace quad_rope_lift
