
#include "bpe_core.hpp"
#include "absl/container/flat_hash_map.h"

namespace script_bpe {
        FastTokenizer::FastTokenizer(const std::unordered_map<char32_t, CharSCRIPTEnc>& char_script_enc,
                                                                 const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules)
                : char_script_enc_(char_script_enc.begin(), char_script_enc.end()),
                    merge_rules_(merge_rules.begin(), merge_rules.end()) {
        whitespace_script_id_ = char_script_enc_.find(U' ')->second.script_id;
        inherited_script_id_ = char_script_enc_.find(U'ー')->second.script_id; // example for Japanese long vowel mark
        auto& inherited_c_script_id_ = char_script_enc_.find(U'\u200d')->second.script_id; // example for zero-width joiner
        auto& han_script_id_ = char_script_enc_.find(U'漢')->second.script_id;
        auto& hiragana_script_id_ = char_script_enc_.find(U'ひ')->second.script_id;
        // re-code hiragana to han and inherited(c) to inherited(lm)
        for (auto& it : char_script_enc_) {
            if (it.second.script_id == hiragana_script_id_) it.second.script_id = han_script_id_;
            if (it.second.script_id == inherited_c_script_id_) it.second.script_id = inherited_script_id_;
        }
        worker_state_ = WorkerState(); // Initialize merge heap and token array
        worker_state_.token_array.resize(4096); // Reserve initial size
    }
    py::array_t<int> FastTokenizer::encode(const std::u32string& text) {
        if (text.empty()) {
            return py::array_t<int>(); // Return empty array for empty input
        }
        size_t start = 0, end = 0;
        int last_script_id = -1;
        size_t required_capacity = text.length() * 2;
        if(worker_state_.token_array.size() < required_capacity) {
            worker_state_.token_array.resize(2 * required_capacity);
        }
        auto& token_array = worker_state_.token_array;

        for(size_t ci = 0; ci < text.length(); ci++) {
            auto it = char_script_enc_.find(text[ci]);
            if (it == char_script_enc_.end()) { // invalid character, skip
                continue;
            }
            auto& script_id = it->second.script_id;
            if(script_id != last_script_id && script_id != inherited_script_id_) { // new pretoken
                if(last_script_id == whitespace_script_id_ && end-start==1) {
                    last_script_id = script_id; // single space, include, but set script id to non-space
                }
                else {
                    apply_bpe_merging(worker_state_, start, end); // apply BPE merging to previous pretoken
                    start = end; // reset start for new pretoken
                    last_script_id = script_id;
                }
            }
            if (it->second.char_token_id == -1) { // still pair, never merged
                token_array[end++] = it->second.block_id;
                token_array[end++] = it->second.index_id;
            }
            else { // has token id, check script and maybe tokenize pretoken
                token_array[end++] = it->second.char_token_id;
            }
        }
        apply_bpe_merging(worker_state_, start, end); // last pretoken
        int final_size = remove_gaps(token_array, end);
        return py::array_t<int>(final_size, token_array.data());
    }

    int FastTokenizer::remove_gaps(std::vector<int>& token_array, int end) {
        int write_pos = 0;
        for (int read_pos = 0; read_pos < end; ++read_pos) {
            if (token_array[read_pos] != -1) {
                token_array[write_pos++] = token_array[read_pos];
            }
        }
        return write_pos; // Return new size
    }

    inline void FastTokenizer::apply_bpe_merging(WorkerState& worker_state, int start, int end) {
        if (end - start < 2) return; // Need at least 2 tokens to merge
        
        // Find all possible merges in this chunk - use consecutive individual tokens
        auto& token_array = worker_state.token_array;
        auto& merge_heap = worker_state.merge_heap;
        for (int i = start; i < end - 1; ++i) {
            // Create pairs for merging: (token1, token2)
            std::pair<int, int> merge_key = {token_array[i], token_array[i+1]};
            auto it = merge_rules_.find(merge_key);
            if (it != merge_rules_.end()) {
                merge_heap.push({
                    it->second.first,     // priority
                    i,                    // from_a
                    token_array[i],       // val_a
                    i + 1,                // from_b
                    token_array[i+1],     // val_b
                    it->second.second     // to_id
                });
            }
        }
        
        // Apply merges in priority order
        while (!merge_heap.empty()) {
            FastTokenizer::MergeItem item = merge_heap.top();
            merge_heap.pop();
            // Verify merge is still valid
            if (token_array[item.from_a] != item.val_a || 
                token_array[item.from_b] != item.val_b) continue;
            
            // Perform merge - replace first token with merged token, mark second token as deleted
            token_array[item.from_a] = item.to_id;
            token_array[item.from_b] = -1;  // Mark as deleted
            
            // Add new potential merges
            find_and_add_new_merges(worker_state, start, end, item.from_a, item.from_b);
        }
    }

    inline void FastTokenizer::find_and_add_new_merges(WorkerState& worker_state, int start, int end, int from_a, int from_b) {
        auto& tokens = worker_state.token_array;
        auto& merge_heap = worker_state.merge_heap;

        // Find next valid token after merge
        int next_pos = from_b + 1;
        while (next_pos < end && tokens[next_pos] == -1) {
            next_pos++;
        }
        
        // Check merge with next token
        if (next_pos < end) {
            std::pair<int, int> merge_key = {tokens[from_a], tokens[next_pos]};
            auto it = merge_rules_.find(merge_key);
            if (it != merge_rules_.end()) {
                merge_heap.push({
                    it->second.first,
                    from_a,
                    tokens[from_a],
                    next_pos,
                    tokens[next_pos],
                    it->second.second
                });
            }
        }
        
        // Find previous valid token before merge
        int prev_pos = from_a - 1;
        while (prev_pos >= start && tokens[prev_pos] == -1) {
            prev_pos--;
        }
        
        // Check merge with previous token
        if (prev_pos >= start) {
            std::pair<int, int> merge_key = {tokens[prev_pos], tokens[from_a]};
            auto it = merge_rules_.find(merge_key);
            if (it != merge_rules_.end()) {
                merge_heap.push({
                    it->second.first,
                    prev_pos,
                    tokens[prev_pos],
                    from_a,
                    tokens[from_a],
                    it->second.second
                });
            }
        }
    }
    

}