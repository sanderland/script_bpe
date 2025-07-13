# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
#cmake_policy(SET CMP0148 NEW)
project(my_fast_tokenizer LANGUAGES CXX)

# Find dependencies
include(FetchContent)

# Find pybind11 and Python
find_package(pybind11 REQUIRED)

# Find Abseil on the system
find_package(absl REQUIRED COMPONENTS hash raw_hash_set strings)

# Add GTL
FetchContent_Declare(
    gtl
    GIT_REPOSITORY https://github.com/greg7mdp/gtl.git
    GIT_TAG        v1.2.0
)

# Add robin_hood
FetchContent_Declare(
    robin_hood
    GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
    GIT_TAG 3.11.5
)

# Make all dependencies available
FetchContent_MakeAvailable(gtl robin_hood)

# Note: It's better to use CMake's check_cxx_compiler_flag for portability
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_HAS_MARCH_NATIVE)

# Function to set common compile options
function(set_common_target_options target implementation)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE 
        -O3 
        -DNDEBUG
        -Wno-interference-size  # Suppress GTL hardware interference size warnings
    )
    if(COMPILER_HAS_MARCH_NATIVE)
        target_compile_options(${target} PRIVATE -march=native)
    endif()

    # Enable parallel LTO
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_target_properties(${target} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
        target_compile_options(${target} PRIVATE -flto=auto)
    endif()

    # Add common includes
    target_include_directories(${target} PRIVATE 
        ${gtl_SOURCE_DIR}/include
        ${robin_hood_SOURCE_DIR}/src/include
    )

    # Set implementation-specific options
    if(implementation STREQUAL "std")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_STD)
    elseif(implementation STREQUAL "absl")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_ABSL)
        target_link_libraries(${target} PRIVATE absl::flat_hash_map)
    elseif(implementation STREQUAL "gtl")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_GTL)
    elseif(implementation STREQUAL "robinhood")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_ROBINHOOD)
    endif()
endfunction()

# Function to create targets for each implementation
function(create_implementation_targets target_base source_files)
    foreach(impl IN ITEMS std absl gtl robinhood)
        set(target_name "${target_base}_${impl}")
        if(target_base STREQUAL "fast_tokenizer_cpp")
            pybind11_add_module(${target_name} ${source_files})
            set_target_properties(${target_name} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/script_bpe/bpe/fast"
                OUTPUT_NAME "${target_name}"
            )
        else()
            add_executable(${target_name} ${source_files})
            target_compile_definitions(${target_name} PRIVATE NO_PYTHON_BINDINGS)
        endif()
        set_common_target_options(${target_name} ${impl})
    endforeach()
endfunction()

# Create all variants of the Python module
create_implementation_targets(
    fast_tokenizer_cpp
    "script_bpe/bpe/fast/python_bindings.cpp;script_bpe/bpe/fast/bpe_core.cpp"
)

# Create all variants of the benchmark executable
create_implementation_targets(
    benchmark_cpp_fast
    "script_bpe/bpe/fast/benchmark_cpp_fast.cpp;script_bpe/bpe/fast/bpe_core.cpp"
)