#pragma once

#include <vector>
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <cstdint>
#include <tuple>

#ifdef USE_ROBIN_HOOD
#include "robin_hood.h"
#else
#include "gtl/phmap.hpp"
#endif

#ifndef NO_PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
typedef py::array_t<int> encode_return_t;
#else
typedef std::vector<int> encode_return_t;
#endif


namespace std {
    template<>
    struct hash<pair<int, int>> {
        // A higher-quality hash combiner
        size_t operator()(const pair<int, int>& p) const {
            size_t h1 = hash<int>{}(p.first);
            size_t h2 = hash<int>{}(p.second);

            // A good way to combine hashes to minimize collisions, from Boost
            // See https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
}
inline uint64_t pack_key(int a, int b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
}

namespace script_bpe {
    struct CharSCRIPTEnc {
        int script_id;
        int block_id;
        int index_id;
        int char_token_id; // -1 if character is raw pair
    };
    // Priority queue item for BPE merging (defined early for use in functions)
    struct MergeItem {
        int priority;
        int from_a;
        int val_a;
        int from_b;
        int val_b;
        int to_id;
        
        bool operator<(const MergeItem& other) const { // min heap
            return std::tie(priority, from_a) > std::tie(other.priority, other.from_a);
        }
    };

    struct WorkerState {
        std::priority_queue<MergeItem> merge_heap;
        std::vector<int> token_array;
    };  

    class FastTokenizer {
    public:
        // Constructor
        FastTokenizer(const std::vector<CharSCRIPTEnc>& char_script_enc,
             const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules);
        encode_return_t encode(const std::u32string& text);
        
    private:
        // Core data structures
        std::vector<CharSCRIPTEnc> char_script_enc_;
#ifdef USE_ROBIN_HOOD
        robin_hood::unordered_flat_map<uint64_t, std::pair<int, int>> merge_rules_;
#else
        gtl::flat_hash_map<uint64_t, std::pair<int, int>> merge_rules_;
#endif
        int whitespace_script_id_, inherited_script_id_;
        WorkerState worker_state_;

        // Core processing functions
        void try_push_merge(std::priority_queue<MergeItem>& merge_heap,
                          int a, int b, const std::vector<int>& token_array);
        void apply_bpe_merging(WorkerState& worker_state, int start, int end);
        int remove_gaps(std::vector<int>& token_array, int end);
    };
}