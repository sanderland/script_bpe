#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <iostream>
#include "bpe_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_tokenizer_cpp, m) {
    m.doc() = "Fast C++ SCRIPT tokenizer implementation";
    
    // Expose CharSCRIPTEnc struct to Python
    py::class_<script_bpe::CharSCRIPTEnc>(m, "CharSCRIPTEnc")
        .def(py::init<int, int, int, int>())
        .def_readwrite("script_id", &script_bpe::CharSCRIPTEnc::script_id)
        .def_readwrite("block_id", &script_bpe::CharSCRIPTEnc::block_id)
        .def_readwrite("index_id", &script_bpe::CharSCRIPTEnc::index_id)
        .def_readwrite("char_token_id", &script_bpe::CharSCRIPTEnc::char_token_id);
    
    py::class_<script_bpe::FastTokenizer>(m, "FastTokenizer")
        .def(py::init<const std::unordered_map<char32_t, script_bpe::CharSCRIPTEnc>&,
                     const std::unordered_map<std::pair<int, int>, std::pair<int, int>>&>(),
              py::arg("char_script_enc"), py::arg("merge_rules"))
        .def("encode", &script_bpe::FastTokenizer::encode);
}