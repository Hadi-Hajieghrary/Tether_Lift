/// @file test_layer4_eso.cc
/// @brief Unit tests for Layer 4 (Extended State Observer).

#include <gtest/gtest.h>
#include "control/layer4_eso.h"

namespace gpac {
namespace {

TEST(ESOTest, ConstructsSuccessfully) {
  Layer4Params params;
  params.omega_o = 100.0;
  params.b0 = 1.0;
  params.max_disturbance = 10.0;

  EXPECT_NO_THROW({
    ExtendedStateObserver eso(params);
  });
}

TEST(ESOTest, HasCorrectPorts) {
  Layer4Params params;
  ExtendedStateObserver eso(params);

  EXPECT_GT(eso.num_input_ports(), 0);
  EXPECT_GT(eso.num_output_ports(), 0);
}

TEST(ESOTest, CreateContextSucceeds) {
  Layer4Params params;
  ExtendedStateObserver eso(params);

  auto context = eso.CreateDefaultContext();
  EXPECT_NE(context, nullptr);
}

TEST(ESOTest, InitialStateZero) {
  Layer4Params params;
  ExtendedStateObserver eso(params);
  auto context = eso.CreateDefaultContext();

  const auto& state = context->get_continuous_state_vector();
  EXPECT_EQ(state.size(), 9);  // 3 states x 3 axes

  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(state[i], 0.0, 1e-10);
  }
}

}  // namespace
}  // namespace gpac
