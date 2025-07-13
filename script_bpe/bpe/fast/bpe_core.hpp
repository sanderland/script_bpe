#pragma once

#include <vector>
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <cstdint>
#include <tuple>

// All map-specific configuration in a single block
#ifdef MERGE_MAP_STD
    #include <unordered_map>
    #define MERGE_MAP_NAME "std::unordered_map"
    namespace script_bpe {
        using merge_map_t = std::unordered_map<uint64_t, std::pair<int, int>>;
    }
#elif defined(MERGE_MAP_ABSL)
    #include "absl/container/flat_hash_map.h"
    #define MERGE_MAP_NAME "absl::flat_hash_map"
    namespace script_bpe {
        using merge_map_t = absl::flat_hash_map<uint64_t, std::pair<int, int>>;
    }
#elif defined(MERGE_MAP_GTL)
    #include "gtl/phmap.hpp"
    #define MERGE_MAP_NAME "gtl::flat_hash_map"
    namespace script_bpe {
        using merge_map_t = gtl::flat_hash_map<uint64_t, std::pair<int, int>>;
    }
#elif defined(MERGE_MAP_ROBINHOOD)
    #include "robin_hood.h"
    #define MERGE_MAP_NAME "robin_hood::unordered_flat_map"
    namespace script_bpe {
        using merge_map_t = robin_hood::unordered_flat_map<uint64_t, std::pair<int, int>>;
    }
#else
    #error "One of MERGE_MAP_STD, MERGE_MAP_ABSL, MERGE_MAP_GTL, or MERGE_MAP_ROBINHOOD must be defined"
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
        size_t operator()(const pair<int, int>& p) const {
            size_t h1 = hash<int>{}(p.first);
            size_t h2 = hash<int>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
}

namespace script_bpe {
    // Helper function to pack two integers into a uint64_t key
    inline uint64_t pack_key(int a, int b) {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
    }

    struct CharSCRIPTEnc {
        int script_id;
        int block_id;
        int index_id;
        int char_token_id; // -1 if character is raw pair
    };

    struct MergeItem {
        int priority;
        int from_a;
        int val_a;
        int from_b;
        int val_b;
        int to_id;
        
        bool operator<(const MergeItem& other) const {
            return std::tie(priority, from_a) > std::tie(other.priority, other.from_a);
        }
    };

    struct WorkerState {
        std::priority_queue<MergeItem> merge_heap;
        std::vector<int> token_array;
    };

    class FastTokenizer {
    public:
        FastTokenizer(const std::vector<CharSCRIPTEnc>& char_script_enc,
                     const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules);
        encode_return_t encode(const std::u32string& text);
        
    private:
        std::vector<CharSCRIPTEnc> char_script_enc_;
        merge_map_t merge_rules_;
        int whitespace_script_id_, inherited_script_id_;
        WorkerState worker_state_;

        void try_push_merge(std::priority_queue<MergeItem>& merge_heap,
                          int a, int b, const std::vector<int>& token_array);
        void apply_bpe_merging(WorkerState& worker_state, int start, int end);
        int remove_gaps(std::vector<int>& token_array, int end);
    };
}
