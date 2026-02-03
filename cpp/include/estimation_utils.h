#pragma once

#include <Eigen/Core>
#include <vector>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/leaf_system.h>

namespace quad_rope_lift {

/// Extracts attachment positions of multiple bodies from the plant state.
///
/// This system takes the full plant state and outputs a stacked vector of
/// world-frame positions for specified body attachment points.
///
/// Input:
///   - plant_state: Full state vector from MultibodyPlant
///
/// Output:
///   - positions: Stacked vector [x0,y0,z0, x1,y1,z1, ...]
///
class AttachmentPositionExtractor final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AttachmentPositionExtractor);

  /// Constructs the extractor.
  ///
  /// @param plant The MultibodyPlant
  /// @param bodies Bodies to extract positions from
  /// @param offsets Local frame offset for each body's attachment point
  AttachmentPositionExtractor(
      const drake::multibody::MultibodyPlant<double>& plant,
      const std::vector<const drake::multibody::RigidBody<double>*>& bodies,
      const std::vector<Eigen::Vector3d>& offsets);

  const drake::systems::InputPort<double>& get_plant_state_input_port() const {
    return get_input_port(plant_state_port_);
  }

  const drake::systems::OutputPort<double>& get_positions_output_port() const {
    return get_output_port(positions_port_);
  }

 private:
  void CalcPositions(const drake::systems::Context<double>& context,
                     drake::systems::BasicVector<double>* output) const;

  const drake::multibody::MultibodyPlant<double>& plant_;
  std::vector<drake::multibody::BodyIndex> body_indices_;
  std::vector<Eigen::Vector3d> offsets_;
  int plant_state_port_{-1};
  int positions_port_{-1};
};

/// Simple system that outputs constant cable lengths.
///
/// This is a placeholder - in a real system, cable lengths would be estimated.
class CableLengthSource final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CableLengthSource);

  explicit CableLengthSource(const std::vector<double>& lengths);

  const drake::systems::OutputPort<double>& get_lengths_output_port() const {
    return get_output_port(0);
  }

 private:
  void CalcLengths(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output) const;

  std::vector<double> lengths_;
};

/// Aggregates tension magnitudes from multiple rope systems.
///
/// Input ports: N tension vectors [T, fx, fy, fz] each
/// Output: Single vector of tension magnitudes [T0, T1, ..., T_{N-1}]
class TensionAggregator final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TensionAggregator);

  explicit TensionAggregator(int num_cables);

  const drake::systems::InputPort<double>& get_tension_input_port(int index) const {
    return get_input_port(index);
  }

  const drake::systems::OutputPort<double>& get_tensions_output_port() const {
    return get_output_port(0);
  }

 private:
  void CalcTensions(const drake::systems::Context<double>& context,
                    drake::systems::BasicVector<double>* output) const;

  int num_cables_;
};

}  // namespace quad_rope_lift
