#pragma once

/// @file load_trajectory_generator.h
/// @brief Load-centric trajectory generation for cooperative transport.
///
/// Generates smooth, dynamically feasible trajectories for the load,
/// respecting cable constraints and formation geometry. Trajectories
/// are defined in the load frame, and each quadcopter computes its
/// own reference by adding formation offset.

#include <Eigen/Core>
#include <vector>

#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/framework/basic_vector.h>

namespace quad_rope_lift {

/// A 3D position waypoint with timing and smoothness parameters.
struct LoadWaypoint {
  Eigen::Vector3d position{0, 0, 0};  ///< Target load position [m]
  double arrival_time = 0.0;           ///< Time to arrive at this point [s]
  double hold_time = 0.0;              ///< Time to hold before moving [s]

  /// Maximum velocity for approach (0 = use global default)
  double max_velocity = 0.0;

  /// Maximum acceleration for approach (0 = use global default)
  double max_acceleration = 0.0;
};

/// Parameters for the load trajectory generator.
struct LoadTrajectoryParams {
  // Default dynamics limits
  double max_velocity = 1.0;          ///< Maximum load velocity [m/s]
  double max_acceleration = 2.0;      ///< Maximum load acceleration [m/s²]
  double max_jerk = 5.0;              ///< Maximum load jerk [m/s³]

  // Altitude constraints
  double min_altitude = 0.5;          ///< Minimum load altitude [m]
  double max_altitude = 10.0;         ///< Maximum load altitude [m]

  // Approach parameters
  double landing_velocity = 0.2;      ///< Max velocity for landing [m/s]
  double landing_acceleration = 0.5;  ///< Max acceleration for landing [m/s²]

  // Smoothing parameters
  double corner_radius = 0.3;         ///< Radius for smoothing waypoint corners [m]
  double blend_time = 0.5;            ///< Time constant for trajectory blending [s]

  // Formation awareness (optional - helps plan feasible trajectories)
  double estimated_cable_length = 1.5;///< Expected cable length [m]
  int estimated_num_quads = 4;        ///< Expected number of quadcopters

  // Safety margins
  double altitude_safety_margin = 0.2;///< Extra margin above min altitude [m]
};

/// Trajectory segment type for internal planning.
enum class SegmentType {
  kHold,        ///< Stationary at waypoint
  kLinear,      ///< Linear interpolation (for slow moves)
  kMinJerk,     ///< Minimum-jerk polynomial
  kTrapezoidal  ///< Trapezoidal velocity profile (accel-cruise-decel)
};

/// Load-centric trajectory generator.
///
/// This system generates reference trajectories for the suspended load.
/// Each quadcopter then adds its formation offset to compute its own
/// reference position. This ensures:
///   1. Load is the "virtual leader" - trajectory is defined for load
///   2. Formation geometry is maintained automatically
///   3. Trajectories respect load dynamics constraints
///
/// The generator supports multiple trajectory types:
///   - Waypoint sequences with smooth interpolation
///   - Simple altitude ramps for lift/descend
///   - Polynomial (minimum-jerk) trajectories
///   - External reference tracking
///
/// Output ports:
///   - load_position_des: Desired load position [m] (3)
///   - load_velocity_des: Desired load velocity [m/s] (3)
///   - load_acceleration_des: Desired load acceleration [m/s²] (3)
///   - trajectory_complete: 1.0 if trajectory finished, 0.0 otherwise (1)
///
class LoadTrajectoryGenerator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LoadTrajectoryGenerator);

  /// Constructs a waypoint-based trajectory generator.
  ///
  /// @param waypoints Sequence of waypoints to visit
  /// @param params Trajectory parameters
  LoadTrajectoryGenerator(
      const std::vector<LoadWaypoint>& waypoints,
      const LoadTrajectoryParams& params = LoadTrajectoryParams());

  /// Constructs a simple altitude ramp trajectory generator.
  ///
  /// @param initial_pos Initial load position [m]
  /// @param final_pos Final load position [m]
  /// @param start_time Time to begin motion [s]
  /// @param params Trajectory parameters
  static LoadTrajectoryGenerator CreateAltitudeRamp(
      const Eigen::Vector3d& initial_pos,
      const Eigen::Vector3d& final_pos,
      double start_time,
      const LoadTrajectoryParams& params = LoadTrajectoryParams());

  /// Constructs a hovering (stationary) trajectory generator.
  ///
  /// @param hover_position Fixed position to maintain [m]
  static LoadTrajectoryGenerator CreateHover(
      const Eigen::Vector3d& hover_position);

  // === Output port accessors ===
  const drake::systems::OutputPort<double>& get_position_output_port() const {
    return get_output_port(position_port_);
  }
  const drake::systems::OutputPort<double>& get_velocity_output_port() const {
    return get_output_port(velocity_port_);
  }
  const drake::systems::OutputPort<double>& get_acceleration_output_port() const {
    return get_output_port(acceleration_port_);
  }
  const drake::systems::OutputPort<double>& get_complete_output_port() const {
    return get_output_port(complete_port_);
  }

  /// Get waypoints.
  const std::vector<LoadWaypoint>& waypoints() const { return waypoints_; }

  /// Get parameters.
  const LoadTrajectoryParams& params() const { return params_; }

 private:
  // Constructor helper
  void Initialize();

  // Output calculations
  void CalcPosition(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;
  void CalcVelocity(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;
  void CalcAcceleration(const drake::systems::Context<double>& context,
                        drake::systems::BasicVector<double>* output) const;
  void CalcComplete(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;

  // Trajectory evaluation
  void EvaluateTrajectory(double t,
                          Eigen::Vector3d& position,
                          Eigen::Vector3d& velocity,
                          Eigen::Vector3d& acceleration) const;

  // Minimum-jerk polynomial coefficients for one axis
  // p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
  struct MinJerkCoeffs {
    double a0, a1, a2, a3, a4, a5;
    double duration;

    void Evaluate(double t, double& pos, double& vel, double& accel) const;
  };

  // Compute minimum-jerk coefficients for a segment
  MinJerkCoeffs ComputeMinJerkSegment(
      double p0, double v0, double a0,
      double pf, double vf, double af,
      double duration) const;

  // Internal trajectory segment
  struct TrajectorySegment {
    double start_time;
    double end_time;
    SegmentType type;
    Eigen::Vector3d start_pos;
    Eigen::Vector3d end_pos;
    Eigen::Vector3d start_vel;
    Eigen::Vector3d end_vel;
    std::array<MinJerkCoeffs, 3> min_jerk_coeffs;  // For kMinJerk type
  };

  // Build trajectory segments from waypoints
  void BuildTrajectorySegments();

  // Find segment for given time
  int FindSegmentIndex(double t) const;

  // Parameters
  LoadTrajectoryParams params_;

  // Waypoints
  std::vector<LoadWaypoint> waypoints_;

  // Precomputed trajectory segments
  std::vector<TrajectorySegment> segments_;

  // Total trajectory duration
  double total_duration_;

  // Port indices
  drake::systems::OutputPortIndex position_port_;
  drake::systems::OutputPortIndex velocity_port_;
  drake::systems::OutputPortIndex acceleration_port_;
  drake::systems::OutputPortIndex complete_port_;
};

}  // namespace quad_rope_lift
