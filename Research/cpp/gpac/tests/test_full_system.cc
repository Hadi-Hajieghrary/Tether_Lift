/// @file test_full_system.cc
/// @brief Integration tests for the complete GPAC system.

#include <gtest/gtest.h>
#include "control/gpac_controller.h"
#include "utils/parameters.h"

namespace gpac {
namespace {

TEST(FullSystemTest, GpacControllerConstructs) {
  GpacParams params;  // Uses default values

  EXPECT_NO_THROW({
    GpacController controller(0, 3, params);  // Drone 0 of 3
  });
}

TEST(FullSystemTest, GpacControllerHasPorts) {
  GpacParams params;
  GpacController controller(0, 3, params);

  EXPECT_GT(controller.num_input_ports(), 0);
  EXPECT_GT(controller.num_output_ports(), 0);
}

TEST(FullSystemTest, GpacControllerCreatesContext) {
  GpacParams params;
  GpacController controller(0, 3, params);

  auto context = controller.CreateDefaultContext();
  EXPECT_NE(context, nullptr);
}

TEST(FullSystemTest, MultipleControllersIndependent) {
  GpacParams params;

  GpacController controller0(0, 3, params);
  GpacController controller1(1, 3, params);
  GpacController controller2(2, 3, params);

  auto context0 = controller0.CreateDefaultContext();
  auto context1 = controller1.CreateDefaultContext();
  auto context2 = controller2.CreateDefaultContext();

  EXPECT_NE(context0, nullptr);
  EXPECT_NE(context1, nullptr);
  EXPECT_NE(context2, nullptr);
}

TEST(FullSystemTest, DefaultParametersReasonable) {
  GpacParams params;

  // Layer 1 gains
  EXPECT_GT(params.layer1.Kp.sum(), 0);
  EXPECT_GT(params.layer1.k_q, 0);

  // Layer 2 gains
  EXPECT_GT(params.layer2.k_R, 0);
  EXPECT_GT(params.layer2.k_Ω, 0);

  // Layer 4 observer bandwidth
  EXPECT_GT(params.layer4.omega_o, 0);

  // Physical parameters
  EXPECT_GT(params.quadrotor.mass, 0);
  EXPECT_GT(params.quadrotor.cable_length, 0);
}

}  // namespace
}  // namespace gpac
