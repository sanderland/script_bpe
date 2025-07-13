#pragma once

#include <vector>
#include "absl/container/flat_hash_map.h"
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <cstdint>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


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

      // Helper to check merge_rules and push to heap if found
    inline void try_push_merge(std::priority_queue<MergeItem>& merge_heap,
                               const absl::flat_hash_map<uint64_t, std::pair<int, int>>& merge_rules_,
                               int a, int b, const std::vector<int>& token_array) {
        uint64_t merge_key = pack_key(token_array[a], token_array[b]);
        auto it = merge_rules_.find(merge_key);
        if (it != merge_rules_.end()) {
            merge_heap.push({
                it->second.first,
                a,
                token_array[a],
                b,
                token_array[b],
                it->second.second
            });
        }
    }
    class FastTokenizer {
    public:
        // Constructor
    FastTokenizer(const std::vector<CharSCRIPTEnc>& char_script_enc,
             const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules);
        py::array_t<int> encode(const std::u32string& text);
        
    private:
        // Core data structures
        std::vector<CharSCRIPTEnc> char_script_enc_;
        absl::flat_hash_map<uint64_t, std::pair<int, int>> merge_rules_;
        int whitespace_script_id_, inherited_script_id_;
        WorkerState worker_state_;
        // Core processing functions
        void apply_bpe_merging(WorkerState& worker_state, int start, int end);
        int remove_gaps(std::vector<int>& token_array, int end);
    };
}