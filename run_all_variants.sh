#!/bin/bash

# fail on errors but allow no matches in globs
set -euo pipefail

# Store the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Arrays to store results - declare with indices to avoid unbound variable errors
declare -a variants=(std absl gtl robinhood)
declare -a seconds=()
declare -a tokens_per_sec=()
declare -a cycles_per_token=()

# Initialize arrays with zeros
for i in "${!variants[@]}"; do
    seconds[$i]=0
    tokens_per_sec[$i]=0
    cycles_per_token[$i]=0
done

# Function to parse simple space-separated output format
parse_results() {
    local idx=$1
    local output=$2
    
    # Expect output in format: "BENCHMARK_RESULT seconds tokens_per_sec cycles_per_token"
    if ! read -r tag sec tok cyc <<< $(echo "$output" | grep "^BENCHMARK_RESULT"); then
        echo "Failed to parse benchmark output"
        exit 1
    fi
    
    seconds[$idx]=$sec
    tokens_per_sec[$idx]=$tok
    cycles_per_token[$idx]=$cyc
}

# Clean everything
rm -rf "${BUILD_DIR}" "${SCRIPT_DIR}/CMakeCache.txt" "${SCRIPT_DIR}/CMakeFiles"
mkdir -p "${BUILD_DIR}"

# Configure and build all variants
echo "Configuring and building..."
(
    cd "${BUILD_DIR}" && \
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" && \
    cmake --build "${BUILD_DIR}" -j$(nproc)
) || {
    echo "Build failed!"
    exit 1
}

# Function to run all variants
run_variants() {
    echo "Running all benchmark variants in parallel..."
    cd "${BUILD_DIR}"
    
    # Create temporary files to store outputs
    declare -a tmpfiles=()
    for variant in "${variants[@]}"; do
        tmpfiles+=("$(mktemp)")
    done
    
    # Run all variants in parallel
    echo "Starting benchmarks..."
    for idx in "${!variants[@]}"; do
        ./benchmark_cpp_fast_"${variants[$idx]}" > "${tmpfiles[$idx]}" 2>&1 &
    done
    
    # Wait for all benchmarks to complete
    wait
    
    # Process results in the main shell
    for idx in "${!variants[@]}"; do
        echo "----------------------------------------"
        echo "Results for ${variants[$idx]}:"
        output=$(<"${tmpfiles[$idx]}")
        echo "$output"
        parse_results $idx "$output"
        rm "${tmpfiles[$idx]}"
        echo "----------------------------------------"
    done
    
    cd - > /dev/null
}

# Function to print results table
print_results_table() {
    echo
    echo "Performance Summary"
    echo "----------------------------------------"
    printf "%-10s %12s %15s %12s\n" "Variant" "Time (s)" "MTok/s" "Cycles/tok"
    echo "----------------------------------------"
    for idx in "${!variants[@]}"; do
        mtps=$(echo "${tokens_per_sec[$idx]} / 1000000" | bc -l)
        printf "%-10s %12.3f %15.3f %12.3f\n" \
            "${variants[$idx]}" \
            "${seconds[$idx]}" \
            "$mtps" \
            "${cycles_per_token[$idx]}"
    done
    echo "----------------------------------------"
}

# Run all variants
run_variants

# Print summary table
print_results_table

echo "All variants have been run!"
