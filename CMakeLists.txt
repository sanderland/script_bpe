# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(my_fast_tokenizer LANGUAGES CXX)

# Find pybind11 and Python
find_package(pybind11 REQUIRED)
# This will find Abseil on the system (e.g., from `brew install abseil`, `apt install libabsl-dev`)
# Users might need to set CMAKE_PREFIX_PATH if it's in a non-standard location.
find_package(absl REQUIRED COMPONENTS hash raw_hash_set strings)


# Build the extension as 'fast_tokenizer_cpp' and place it in the package subdirectory
pybind11_add_module(
    fast_tokenizer_cpp
    script_bpe/bpe/fast/python_bindings.cpp
    script_bpe/bpe/fast/bpe_core.cpp
)

# Set the output directory to the package subdirectory
set_target_properties(fast_tokenizer_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/script_bpe/bpe/fast"
    OUTPUT_NAME "fast_tokenizer_cpp"
)

# Link against pybind11 and Abseil libraries
target_link_libraries(
    fast_tokenizer_cpp
    PRIVATE
    pybind11::module
    absl::flat_hash_map
)

# Use C++17
target_compile_features(fast_tokenizer_cpp PRIVATE cxx_std_17)

# Optimization flags
target_compile_options(fast_tokenizer_cpp PRIVATE -O3 -DNDEBUG)
# Note: It's better to use CMake's check_cxx_compiler_flag for portability
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_HAS_MARCH_NATIVE)
if(COMPILER_HAS_MARCH_NATIVE)
    target_compile_options(fast_tokenizer_cpp PRIVATE -march=native)
endif()