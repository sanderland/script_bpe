#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <cstdint>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Custom hash function for std::pair<int, int> for merge lookup
namespace std {
    template<>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            auto h1 = hash<int>{}(p.first);
            auto h2 = hash<int>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };
}

namespace script_bpe {
    struct CharSCRIPTEnc {
        int script_id;
        int block_id;
        int index_id;
        int char_token_id; // -1 if character is raw pair
    };

    class FastTokenizer {
    private:
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
            std::priority_queue<FastTokenizer::MergeItem> merge_heap;
            std::vector<int> token_array;
        };        

    public:
        // Constructor
        FastTokenizer(const std::unordered_map<char32_t, CharSCRIPTEnc>& char_script_enc,
                     const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules);
        
        py::array_t<int> encode(const std::u32string& text);
        
    private:
        // Core data structures
        std::unordered_map<char32_t, CharSCRIPTEnc> char_script_enc_;
        std::unordered_map<std::pair<int, int>, std::pair<int, int>> merge_rules_;
        int whitespace_script_id_, inherited_lm_script_id_, inherited_c_script_id_, han_script_id_, hiragana_script_id_;
        WorkerState worker_state_;
        // Core processing functions
        void apply_bpe_merging(WorkerState& worker_state, int start, int end);
        void find_and_add_new_merges(WorkerState& worker_state, int start, int end, int from_a, int from_b);
        int remove_gaps(std::vector<int>& token_array, int end);
    };
}