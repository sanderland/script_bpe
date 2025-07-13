#!/bin/bash

# Arrays to store results
seconds=()
tokens_per_sec=()
cycles_per_token=()
variants=(std absl gtl robinhood)

# Function to parse JSON-like output and store results
parse_results() {
    local idx=$1
    local output=$2
    
    # Extract values using grep and sed
    seconds[$idx]=$(echo "$output" | grep -o '"seconds":[0-9.]*' | sed 's/"seconds"://')
    tokens_per_sec[$idx]=$(echo "$output" | grep -o '"tokens_per_sec":[0-9.]*' | sed 's/"tokens_per_sec"://')
    cycles_per_token[$idx]=$(echo "$output" | grep -o '"cycles_per_token":[0-9.]*' | sed 's/"cycles_per_token"://')
}

# Function to run all variants
run_variants() {
    echo "Running all benchmark variants..."
    cd build
    idx=0
    for variant in "${variants[@]}"; do
        echo "----------------------------------------"
        echo "Running benchmark_cpp_fast_$variant..."
        output=$(./benchmark_cpp_fast_$variant)
        parse_results $idx "$output"
        echo "$output"
        echo "----------------------------------------"
        ((idx++))
    done
    cd ..
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

# Create build directory if it doesn't exist
mkdir -p build

# Configure and build all variants
cd build
cmake ..
make -j$(nproc)
cd ..

# Run all variants
run_variants

# Print summary table
print_results_table

echo "All variants have been run!"
