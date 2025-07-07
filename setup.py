from setuptools import setup, Extension
import pybind11

# Define your extension module
ext_modules = [
    Extension(
        "script_bpe.bpe.fast.fast_tokenizer_cpp",
        sources=[
            "script_bpe/bpe/fast/python_bindings.cpp",
            "script_bpe/bpe/fast/bpe_core.cpp",
        ],
        include_dirs=[ # portable path to the pybind11 headers.
            pybind11.get_include(),
        ],
        language="c++",
        # "-O3" for optimization, "-DNDEBUG" to disable debug assertions
        extra_compile_args=["-O3", "-march=native", "-flto", "-DNDEBUG"],
    ),
]

setup(
    ext_modules=ext_modules,
)