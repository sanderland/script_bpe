#include "bpe_core.hpp"
#include <algorithm>
#include <iostream>
#include <set>

namespace script_bpe {
    FastTokenizer::FastTokenizer(const std::unordered_map<char32_t, CharSCRIPTEnc>& char_script_enc,
                                 const std::unordered_map<std::pair<int, int>, std::pair<int, int>>& merge_rules)
        : char_script_enc_(char_script_enc), merge_rules_(merge_rules) {
    }
    std::vector<int> FastTokenizer::encode(const std::u32string& text) {
        if (text.empty()) {
            return std::vector<int>(); // Return empty vector for empty input
        }
        size_t start = 0, end = 0;
        std::vector<int> token_array(text.length()*2, -1);

        for(size_t ci = 0; ci < text.length(); ci++) {
            auto it = char_script_enc_.find(text[ci]);
            if (it == char_script_enc_.end()) { // invalid character, skip
                continue;
            }
            if (it->second.char_token_id == -1) { // still pair, never merged
                apply_bpe_merging(token_array, start, end); // tokenize last pretoken
                token_array[end++] = it->second.block_id;
                token_array[end++] = it->second.index_id;
                start = end;
            }
            else { // has token id, check script and maybe tokenize pretoken
                token_array[end++] = it->second.char_token_id;
            }
        }
        apply_bpe_merging(token_array, start, end); // last pretoken
        remove_gaps(token_array, end);
        return token_array;
    }

    void FastTokenizer::remove_gaps(std::vector<int>& token_array, int end) {
        int write_pos = 0;
        for (int read_pos = 0; read_pos < end; ++read_pos) {
            if (token_array[read_pos] != -1) token_array[write_pos++] = token_array[read_pos];
        }
        token_array.resize(write_pos);
    }

    void FastTokenizer::apply_bpe_merging(std::vector<int>& token_array, int start, int end) {
        if (end - start < 2) return; // Need at least 2 tokens to merge
        
        // Build priority queue for merges
        std::priority_queue<FastTokenizer::MergeItem> merge_heap;
        
        // Find all possible merges in this chunk - use consecutive individual tokens
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
            find_and_add_new_merges(token_array, item.from_a, merge_heap);
        }
    }

    void FastTokenizer::find_and_add_new_merges(const std::vector<int>& tokens, int merge_pos, 
                                                std::priority_queue<FastTokenizer::MergeItem>& merge_heap) {
        // Find next valid token after merge
        int next_pos = merge_pos + 1;
        while (next_pos < static_cast<int>(tokens.size()) && tokens[next_pos] == -1) {
            next_pos++;
        }
        
        // Check merge with next token
        if (next_pos < static_cast<int>(tokens.size())) {
            std::pair<int, int> merge_key = {tokens[merge_pos], tokens[next_pos]};
            auto it = merge_rules_.find(merge_key);
            if (it != merge_rules_.end()) {
                merge_heap.push({
                    it->second.first,
                    merge_pos,
                    tokens[merge_pos],
                    next_pos,
                    tokens[next_pos],
                    it->second.second
                });
            }
        }
        
        // Find previous valid token before merge
        int prev_pos = merge_pos - 1;
        while (prev_pos >= 0 && tokens[prev_pos] == -1) {
            prev_pos--;
        }
        
        // Check merge with previous token
        if (prev_pos >= 0) {
            std::pair<int, int> merge_key = {tokens[prev_pos], tokens[merge_pos]};
            auto it = merge_rules_.find(merge_key);
            if (it != merge_rules_.end()) {
                merge_heap.push({
                    it->second.first,
                    prev_pos,
                    tokens[prev_pos],
                    merge_pos,
                    tokens[merge_pos],
                    it->second.second
                });
            }
        }
    }
    

}