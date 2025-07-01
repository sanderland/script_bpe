import heapq
import copy
from script_bpe.bpe.tokenizer import BPETokenizer, MergeRule


def apply_scaffold_bpe(tokenizer: BPETokenizer, additional_vocab_size: int, verbose: bool = False) -> BPETokenizer:
    """
    Apply scaffold BPE to reduce vocabulary size by removing low-frequency merged tokens
    without descendants and redistributing their counts to parent tokens.
    
    Args:
        tokenizer: The BPETokenizer to apply scaffold BPE to
        additional_vocab_size: Target number of merge rules to keep
        verbose: Whether to print debug information
        
    Returns:
        Modified BPETokenizer with reduced vocabulary and renumbered tokens
    """
    # Get token counts from metadata (this has the actual statistics)
    if "tokens" not in tokenizer.metadata:
        raise ValueError("Tokenizer metadata must contain token statistics for scaffold BPE")
    
    if len(tokenizer.merge_rules) <= additional_vocab_size:
        # No scaffolding needed
        return tokenizer
    
    metadata_tokens = {t["id"]: t for t in tokenizer.metadata["tokens"]}
    merge_rules = list(tokenizer.merge_rules)
    
    # Build parent-child relationships from merge rules
    token_to_children = {}
    token_to_parents = {}
    
    # Initialize for all tokens (including base tokens)
    for token_id in metadata_tokens:
        token_to_children[token_id] = set()
        token_to_parents[token_id] = set()
    
    # Build relationships from merge rules
    for merge_rule in merge_rules:
        for parent_id in merge_rule.tokens_from:
            token_to_children[parent_id].add(merge_rule.token_to)
            token_to_parents[merge_rule.token_to].add(parent_id)

    def has_descendants(token_id: int) -> bool:
        """Check if token has any descendants in the merge tree"""
        return len(token_to_children[token_id]) > 0

    def get_all_descendants(token_id: int) -> set[int]:
        """Get all descendants of a token recursively"""
        descendants = set()
        for child in token_to_children[token_id]:
            descendants.add(child)
            descendants.update(get_all_descendants(child))
        return descendants

    # Identify merge rules to remove - we need to remove enough to reach the target
    # Strategy: prioritize removing tokens without descendants, but remove more if needed
    merge_rules_to_remove = []
    tokens_to_remove = set()
    
    num_to_remove = len(merge_rules) - additional_vocab_size
    
    # First pass: remove merge rules without descendants (from most recent backwards)
    for i in range(len(merge_rules) - 1, -1, -1):
        if len(merge_rules_to_remove) >= num_to_remove:
            break
            
        merge_rule = merge_rules[i] 
        token_id = merge_rule.token_to
        
        if not has_descendants(token_id):
            merge_rules_to_remove.append(i)
            tokens_to_remove.add(token_id)
            if verbose:
                print(f"Scaffold-BPE: will remove merge rule {i} creating token {token_id} (no descendants)")
    
    # Second pass: if we still need to remove more, remove from the end regardless of descendants
    if len(merge_rules_to_remove) < num_to_remove:
        if verbose:
            print(f"Scaffold-BPE: need to remove {num_to_remove - len(merge_rules_to_remove)} more merge rules")
        
        for i in range(len(merge_rules) - 1, -1, -1):
            if len(merge_rules_to_remove) >= num_to_remove:
                break
                
            if i not in merge_rules_to_remove:
                merge_rule = merge_rules[i]
                token_id = merge_rule.token_to
                merge_rules_to_remove.append(i)
                tokens_to_remove.add(token_id)
                if verbose:
                    print(f"Scaffold-BPE: will remove merge rule {i} creating token {token_id} (forced removal)")
    
    if verbose:
        print(f"Scaffold-BPE: removing {len(merge_rules_to_remove)} merge rules")
    
    # Create updated token counts by redistributing removed token counts to parents
    updated_token_counts = copy.deepcopy(metadata_tokens)
    
    for token_id in tokens_to_remove:
        token_count = updated_token_counts[token_id]["final_count"]
        if verbose:
            print(f"Scaffold-BPE: redistributing count {token_count:,} from removed token {token_id}")
        
        # Find the merge rule that created this token
        merge_rule = None
        for mr in merge_rules:
            if mr.token_to == token_id:
                merge_rule = mr
                break
        
        if merge_rule:
            # Redistribute count equally to parent tokens
            count_per_parent = token_count // len(merge_rule.tokens_from)
            remainder = token_count % len(merge_rule.tokens_from)
            
            for j, parent_id in enumerate(merge_rule.tokens_from):
                additional_count = count_per_parent + (1 if j < remainder else 0)
                updated_token_counts[parent_id]["final_count"] += additional_count
                if verbose:
                    print(f"   +{additional_count:,} to parent token {parent_id}")

    # Filter out removed merge rules
    kept_merge_rules = [mr for i, mr in enumerate(merge_rules) 
                       if i not in merge_rules_to_remove]
    
    if verbose:
        print(f"Scaffold-BPE: kept {len(kept_merge_rules)} merge rules out of {len(merge_rules)}")
    
    # Renumber tokens to close gaps
    # Strategy: keep base tokens as-is, renumber merge tokens sequentially
    old_to_new_id = {}
    
    # Base tokens keep their original IDs
    for token_id in tokenizer.pretokenizer.base_tokens:
        old_to_new_id[token_id] = token_id
    
    # Renumber kept merge tokens sequentially starting after base tokens
    next_token_id = max(tokenizer.pretokenizer.base_tokens) + 1
    for merge_rule in kept_merge_rules:
        if merge_rule.token_to not in old_to_new_id:
            old_to_new_id[merge_rule.token_to] = next_token_id
            next_token_id += 1
    
    # Give removed tokens negative IDs (scaffold tokens)
    scaffold_token_id = -1
    for token_id in tokens_to_remove:
        old_to_new_id[token_id] = scaffold_token_id
        scaffold_token_id -= 1
    
    # Create renumbered merge rules
    renumbered_merge_rules = []
    for merge_rule in kept_merge_rules:
        new_tokens_from = tuple(old_to_new_id[tid] for tid in merge_rule.tokens_from)
        new_token_to = old_to_new_id[merge_rule.token_to]
        renumbered_merge_rules.append(MergeRule(tokens_from=new_tokens_from, token_to=new_token_to))
    
    # Update metadata with renumbered tokens and updated counts
    new_metadata = copy.deepcopy(tokenizer.metadata)
    new_token_metadata = []
    
    for token_id, token_data in updated_token_counts.items():
        if token_id not in tokens_to_remove:  # Only keep non-removed tokens
            new_token_data = copy.deepcopy(token_data)
            new_token_data["id"] = old_to_new_id[token_id]
            new_token_metadata.append(new_token_data)
    
    new_metadata["tokens"] = new_token_metadata
    
    if verbose:
        print(f"Scaffold BPE: {len(tokens_to_remove)} tokens removed and given negative IDs")
        print(f"Scaffold BPE: {len(renumbered_merge_rules)} merge rules remaining")
        print(f"Scaffold BPE: Token IDs renumbered, next available ID: {next_token_id}")
    
    # Create new tokenizer with renumbered merge rules
    return BPETokenizer(
        merge_rules=renumbered_merge_rules,
        pretokenizer=tokenizer.pretokenizer,
        metadata=new_metadata
    )

