#!/bin/bash
# fail on errors but allow no matches in globs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Clean up previous builds
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Configure, build, and run all variants
(
    cd "${BUILD_DIR}" && \
    cmake -S "${SCRIPT_DIR}" -B . && \
    cmake --build . -j$(nproc)
) || { echo "Build failed!"; exit 1; }

# Extract variants from CMakeLists.txt and run them
cd "${BUILD_DIR}"
readarray -t variants < <(grep -o '"__[^"]*"' "${SCRIPT_DIR}/CMakeLists.txt" | tr -d '"')
echo "Detected variants: ${variants[*]}"

# Create temporary files for each variant
declare -a tmpfiles=()
for variant in "${variants[@]}"; do
    tmpfiles+=("$(mktemp)")
done

# Run all variants in parallel
echo "Starting benchmarks..."
for idx in "${!variants[@]}"; do
    ./benchmark_cpp_fast"${variants[$idx]}" > "${tmpfiles[$idx]}" 2>&1 &
done

# Wait for all benchmarks to complete
wait

# Collect all results
all_output=""
for idx in "${!variants[@]}"; do
    output=$(<"${tmpfiles[$idx]}")
    echo "----------------------------------------"
    echo "Results for ${variants[$idx]}:"
    echo "$output"
    all_output+="$output"$'\n'
    rm "${tmpfiles[$idx]}"
    echo "----------------------------------------"
done

# Parse results and print summary table
echo
echo "Performance Summary"
echo "----------------------------------------------------------------------"
printf "%-15s %12s %15s %12s %12s\n" "Variant" "Time (s)" "MTok/s" "Cycles/tok" "% slower"
echo "----------------------------------------------------------------------"

# Read results into an array, then sort and print
best_cycles=$(echo "$all_output" | grep "BENCHMARK_RESULT" | awk '{print $4}' | sort -n | head -1)

echo "$all_output" | grep "BENCHMARK_RESULT" | \
    # Sort by cycles/token (4th field)
    sort -k4 -n | \
    # Format and print each line with variant name
    awk -v variants="${variants[*]}" -v best="$best_cycles" '
    BEGIN {
        split(variants, var_arr, " ")
        var_idx = 1
    }
    {
        mtps = $3 / 1000000;
        pct_slower = ($4 / best - 1.0) * 100;
        printf "%-15s %12.3f %15.3f %12.1f %11.1f%%\n", var_arr[var_idx++], $2, mtps, $4, pct_slower
    }'

echo "----------------------------------------------------------------------"
echo "All variants have been run!"