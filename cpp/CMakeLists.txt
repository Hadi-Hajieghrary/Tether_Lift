cmake_minimum_required(VERSION 3.16)
project(quad_rope_lift CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find Drake
find_package(drake CONFIG REQUIRED)

# Find Eigen (usually comes with Drake, but explicit is better)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Main executable
add_executable(quad_rope_lift
    src/main.cc
    src/rope_force_system.cc
    src/quadcopter_controller.cc
    src/rope_visualizer.cc
    src/rope_utils.cc
    src/tension_plotter.cc
)

target_include_directories(quad_rope_lift PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quad_rope_lift PRIVATE
    drake::drake
    Eigen3::Eigen
)

# Optionally add clang-tidy
option(ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        set_target_properties(quad_rope_lift PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXE}"
        )
    endif()
endif()

# Copy drake_models path info for runtime
# (Drake will search standard paths, but this helps)
message(STATUS "Drake found at: ${drake_DIR}")
