#!/usr/bin/env bash
# =============================================================================
# Monte Carlo runner for GPAC cooperative transport simulation
# =============================================================================
# Runs the simulation with multiple random seeds and collects results.
#
# Usage:
#   ./run_monte_carlo.sh [NUM_SEEDS] [START_SEED]
#
# Example:
#   ./run_monte_carlo.sh 20        # Run seeds 1..20
#   ./run_monte_carlo.sh 10 100    # Run seeds 100..109
#
# Output:
#   outputs/monte_carlo/summary.csv    - Aggregated statistics
#   outputs/monte_carlo/seed_NNN/      - Per-seed log directories

set -euo pipefail

NUM_SEEDS=${1:-20}
START_SEED=${2:-1}
BINARY="$(dirname "$0")/../cpp/build/quad_rope_lift"
MC_DIR="/workspaces/Tether_Lift/outputs/monte_carlo"

# Verify binary exists
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build first: cd Research/cpp/build && cmake .. -DCMAKE_PREFIX_PATH=/opt/drake && make -j\$(nproc)"
    exit 1
fi

mkdir -p "$MC_DIR"

echo "============================================"
echo "Monte Carlo Simulation: $NUM_SEEDS seeds"
echo "  Seeds: $START_SEED .. $((START_SEED + NUM_SEEDS - 1))"
echo "  Binary: $BINARY"
echo "  Output: $MC_DIR"
echo "============================================"

# CSV header
SUMMARY="$MC_DIR/summary.csv"
echo "seed,output_dir,status,runtime_s" > "$SUMMARY"

COMPLETED=0
FAILED=0

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((START_SEED + i))
    echo ""
    echo "--- Seed $SEED ($((i + 1))/$NUM_SEEDS) ---"

    START_TIME=$(date +%s)

    if $BINARY --seed "$SEED" --headless 2>&1 | tail -1; then
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))

        # Find the latest output directory (just created)
        LATEST_DIR=$(ls -td /workspaces/Tether_Lift/outputs/logs/*/ 2>/dev/null | head -1)

        if [ -n "$LATEST_DIR" ]; then
            # Rename to seed-tagged directory
            SEED_DIR="$MC_DIR/seed_$(printf '%03d' $SEED)"
            if [ -d "$SEED_DIR" ]; then rm -rf "$SEED_DIR"; fi
            cp -r "$LATEST_DIR" "$SEED_DIR"
            echo "$SEED,$SEED_DIR,ok,$RUNTIME" >> "$SUMMARY"
            echo "  -> $SEED_DIR (${RUNTIME}s)"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "$SEED,,no_output,$RUNTIME" >> "$SUMMARY"
            FAILED=$((FAILED + 1))
        fi
    else
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        echo "$SEED,,failed,$RUNTIME" >> "$SUMMARY"
        echo "  FAILED (${RUNTIME}s)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "Monte Carlo Complete"
echo "  Completed: $COMPLETED / $NUM_SEEDS"
echo "  Failed:    $FAILED / $NUM_SEEDS"
echo "  Summary:   $SUMMARY"
echo "============================================"

# Generate aggregated figures if Python script supports it
if [ $COMPLETED -gt 0 ]; then
    echo ""
    echo "To generate aggregated figures:"
    echo "  python Research/scripts/generate_figures.py --monte-carlo $MC_DIR --output-dir IEEE_IROS_2026/Figures"
fi
