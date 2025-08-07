import heapq
import logging
import math
from collections import Counter, defaultdict

from scipy.special import digamma
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import BasePretokenizer
from script_bpe.unigram.model import UnigramToken, UnigramModel
from script_bpe.utils import token_array, create_logger

def log_examples(logger, pretokenizer: BasePretokenizer, tokens_with_scores: list[tuple[UnigramToken, float]], score_label="score", n=5):
    tokens_with_scores.sort(key=lambda x: x[1])
    for i, (t, score) in enumerate(tokens_with_scores):
        if len(tokens_with_scores) > 2*n:
            if i == n:
                logger.debug(" │  ├─ ...")
            if n < i < len(tokens_with_scores) - n:
                continue
        list_item = " ├─" if i < len(tokens_with_scores) - 1 else " └─"
        logger.debug(f" │ {list_item} {repr(pretokenizer.decode(t.base_tokens)):25}  {score_label} = {score:10.3g}  base_tokens = {list(t.base_tokens)}")


def make_initial_vocab(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    pretokenizer: BasePretokenizer,
    additional_num_tokens: int,
    max_token_length: int,
) -> list[UnigramToken]:

    base_tokens = {(t,) for t in pretokenizer.base_tokens}
    substring_freq = Counter({t:0 for t in base_tokens})
    for base_token_seq, count in corpus:
        for i in range(len(base_token_seq)): # SentencePiece uses suffix array, this is simpler but more mem intensive
            for j in range(i + 1, min(len(base_token_seq) + 1, i + max_token_length + 1)):
                if pretokenizer.token_allowed(base_token_seq[i:j]): # char boundaries etc enforced here!
                    substring_freq[tuple(base_token_seq[i:j])] += count

    all_tokens = [ (max(freq,1) * len(token), token) for token, freq in substring_freq.items()]
    selected_tokens = heapq.nlargest(
        len(base_tokens) + additional_num_tokens,
        all_tokens,
        key=lambda item: (item[1] in base_tokens, item[0])
    )
    log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
    tokens = [UnigramToken(base_tokens=token_array(base_token_seq), id=i, log_prob=math.log(score) - log_sum_scores, locked=base_token_seq in base_tokens) for i, (score, base_token_seq) in enumerate(selected_tokens)]
    
    logger.info(f"🌱 Selected {len(pretokenizer.base_tokens):,} + {additional_num_tokens:,} = {len(tokens):,} initial tokens from {len(all_tokens):,} candidates")
    logger.debug(f" ├─ Max length: {max_token_length}")
    logger.debug(f" └─ Source: {corpus.name}: {corpus.metadata}")
    return tokens

def run_e_step(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    model: UnigramModel,
) -> tuple[list[float], float, int]:
    """Performs the Expectation step of the EM algorithm for Unigram."""
        
    expected_count = defaultdict(float)
    objective = total_tokens = 0
    total_pretoken_freq = sum(freq for _, freq in corpus)

    if total_pretoken_freq == 0:
        return expected_count, objective, total_tokens

    for base_token_seq, freq in corpus:
        lattice = model.make_lattice(base_token_seq)
        z, token_prob = lattice.calc_marginal()
        if math.isnan(z): # should not happen
            logger.warning(f"⚠️ Warning: NaN likelihood for pretoken {base_token_seq} with freq={freq}.")
            continue
        for token_id, prob in token_prob.items():
            expected_count[token_id] += prob * freq
        viterbi_path, _ = lattice.viterbi()
        num_tokens_in_sentence = len(viterbi_path)
        total_tokens += num_tokens_in_sentence * freq
        objective -= (z * freq) / total_pretoken_freq

    return expected_count, objective, total_tokens

def run_m_step(
    logger: logging.Logger,
    pretokenizer: BasePretokenizer,
    model: UnigramModel,
    expected_count: dict[int, float],
    dp_smoothing: bool,
    k_expected_frequency_threshold: float,
) -> tuple[UnigramModel, int]:
    """Performs the Maximization step of the EM algorithm for Unigram.
        expected_counts: Expected frequency for each token from E-step
        dp_smoothing: If True, use digamma-based sparsity (like SentencePiece).
                      If False, use standard maximum likelihood estimation.
    """
    # Filter infrequent pieces.
    filtered_tokens = [
        t for t in model.tokens
        if expected_count[t.id] >= k_expected_frequency_threshold or t.locked
    ]
    num_removed = len(model.tokens) - len(filtered_tokens)
    if num_removed > 0:
        filtered_ids = {t.id for t in filtered_tokens}
        removed_tokens = [(t, expected_count[t.id]) for t in model.tokens if t.id not in filtered_ids]
        logger.debug(f" ├─ Removed {num_removed} low-frequency tokens below threshold {k_expected_frequency_threshold} - examples:")
        log_examples(logger, pretokenizer, removed_tokens, "expected count")
            
        model = UnigramModel(pretokenizer, filtered_tokens)

    total_freq = sum(expected_count[t.id] for t in model.tokens)
    if dp_smoothing: # SentencePiece-style: digamma transform with implicit alpha=0 for sparsity bias
        log_total = digamma(total_freq)
        for t in model.tokens:
            t.log_prob = digamma(expected_count[t.id]) - log_total
    else: # Standard maximum likelihood estimation
        for t in model.tokens:
            t.log_prob = math.log(expected_count[t.id] / total_freq)

    return model, num_removed

def prune_tokens(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    pretokenizer: BasePretokenizer,
    model: UnigramModel,
    desired_vocab_size: int,
    shrinking_factor: float,
    defensive: bool,
) -> tuple[UnigramModel, int]:
    """
    Prunes tokens based on their importance to the model using Viterbi-based pruning.
    This is a Python port of the C++ PruneSentencePieces function from SentencePiece.
    
    Args:
        model: The UnigramModel containing current tokens
        corpus: PretokenizedCorpus of pretoken frequencies
        desired_vocab_size: Target vocabulary size
        shrinking_factor: Factor to reduce vocabulary by (default: 0.75)
        verbose: Whether to print progress information
        
    Returns:
        List of Token objects after pruning
    """
    # Calculate target size based on vocab size and shrinking factor
    num_non_base_tokens = len(model.tokens) - len(pretokenizer.base_tokens)
    shrink_n = int( num_non_base_tokens * shrinking_factor)
    target_size = max(desired_vocab_size, num_non_base_tokens - shrink_n)

    # Initialize data structures for tracking token pruning
    always_keep = {t.id: True for t in model.tokens}  # Whether each token must be kept TODO: THIS NAME IS CONFUSING AF
    alternatives = {t.id: [] for t in model.tokens}  # Alternative segmentations for each token

    # First, segments the current tokens to know how each token is resegmented if removed
    # To do so, we take the second best segmentation of token[i].
    # alternatives[i] stores the sequence of second best tokens.
    for token in model.tokens:
        if token.locked:  # Skip locked tokens (must be kept)
            always_keep[token.id] = True
            continue
            
        lattice = model.make_lattice(token.base_tokens)
        best_path, _ = lattice.viterbi(allow_single_token=True)
        
        if len(best_path) >= 2:  # Can safely remove this token if its Viterbi path is split
            always_keep[token.id] = False
        else:  # Try to find alternative segmentation without single-token path
            alt_path, _ = lattice.viterbi(allow_single_token=False)
            if alt_path: # Found alternative segmentation
                always_keep[token.id] = True
                alternatives[token.id] = [t.id for t in alt_path]
            else: # No alternative segmentation found, must keep
                always_keep[token.id] = True

    # Second, segments all sentences to compute likelihood
    # with a unigram language model. inverted[i] stores
    # the set of sentence index where the tokens[i] appears.
    token_count = {t.id: 0.0 for t in model.tokens}
    inverted = {t.id: [] for t in model.tokens}
    vsum = sum(count for _, count in corpus)

    for base_token_seq, count in corpus:
        lattice = model.make_lattice(base_token_seq)
        viterbi_path, _ = lattice.viterbi()
        for token in viterbi_path:
            token_count[token.id] += count
            inverted[token.id].append(base_token_seq)

    total_count = sum(token_count)
    log_total = math.log(total_count)
    candidates = []
    new_tokens = []

    # Finally, computes how likely the LM likelihood is reduced if the token[i] is removed from the vocabulary.
    # Since the exact computation of loss is difficult, we compute the loss approximately by assuming that all
    # token[id] in the sentences are replaced with alternatives[i] when token[i] is removed.
    unused_tokens = []
    for token in model.tokens:
        # not found in Viterbi path. Can remove this entry safely.        
        if (token_count[token.id] == 0 or not always_keep[token.id]) and not token.locked:
            unused_tokens.append(token)
    unused_token_ids = {t.id for t in unused_tokens}

    for token in model.tokens:
        if token.id in unused_token_ids:
            continue
        elif not alternatives[token.id] or token.locked: # no alternatives. Keeps this entry.
            new_tokens.append(token)
        else:  
            # The logprob with the token[i] = log(count[i] / total_count)
            logprob_token = math.log(token_count[token.id]) - log_total
            
            # After removing the token[i], its frequency freq[i] is re-assigned to alternatives.
            # new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
            #         = current_sum + freq[i] * (alternatives[i] - 1)
            logsum_alt = math.log(total_count + token_count[token.id] * (len(alternatives[token.id]) - 1))
            
            # The frequencies of alternatives are increased by freq[i]
            logprob_alt = sum(
                math.log(token_count[alt_id] + token_count[token.id]) - logsum_alt
                for alt_id in alternatives[token.id]
            )
            # The frequency of token[i] = sum of pretoken freqs where token[i] appears
            token_i_freq = sum(count for pretoken, count in corpus if pretoken in inverted[token.id]) / vsum            
            # loss: the diff of likelihood after removing the token[i]
            loss = token_i_freq * (logprob_token - logprob_alt)
            # (NEW FEATURE) if alternatives are already gone, optionally prevent removing this token
            defended = any(tid in unused_token_ids for tid in alternatives[token.id])
            candidates.append((token, loss, defended))

    # reduce vocabulary to target_size
    candidates.sort(key=lambda x: -x[1])
    
    defended_tokens = []
    for token, loss, defended in candidates:
        if len(new_tokens) < target_size:
            new_tokens.append(token)
        elif defensive and defended:
            defended_tokens.append((token, loss))
            new_tokens.append(token)

    pruned_tokens = [(token, loss) for token, loss, _ in candidates if token not in new_tokens]
    

    logger.info(f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(new_tokens):,}")
    logger.debug(f" ├─ Target size: {target_size:,} based on shrinking factor {shrinking_factor} * non base tokens {num_non_base_tokens:,} and desired vocab size {desired_vocab_size}")    
    logger.debug(f" ├─ Dropped {len(unused_tokens):,} tokens not in any optimal path")
    if unused_tokens:
        log_examples(logger, pretokenizer, [(t, t.log_prob) for t in unused_tokens], "logprob")
    logger.debug(f" ├─ Kept {len(new_tokens)} locked tokens")
    if defended_tokens:
        logger.debug(f" ├─ Defended {len(defended_tokens):,} tokens from being removed along with their alternatives")
        log_examples(logger, pretokenizer, defended_tokens, "loss")

    logger.debug(f" ├─ Pruned {len(pruned_tokens):,} tokens from {len(candidates):,} candidates")
    if pruned_tokens:
        log_examples(logger, pretokenizer, pruned_tokens, "loss")
    if candidates:
        logger.debug(f" └─ Candidates loss range: {candidates[0][1]:.4g} to {candidates[-1][1]:.4g}")
    else:
        logger.info(" └─ No candidates for pruning!")
    
    return UnigramModel(model.pretokenizer, new_tokens), len(unused_tokens), len(pruned_tokens), defended_tokens

def finalize_tokens(
    logger: logging.Logger,
    pretokenizer: BasePretokenizer,
    model: UnigramModel,
    vocab_size: int,
) -> tuple[UnigramModel, int]:
    """Finalizes the vocabulary by ensuring required characters are included and keeping top pieces.
    
    Args:
        logger: Logger instance for output
        pretokenizer: The pretokenizer used for tokenization
        model: The UnigramModel containing current tokens
        vocab_size: Target vocabulary size
        
    Returns:
        Tuple of (finalized model, number of tokens removed)
    """
    logger.info(f"✨ Finalizing vocabulary from {len(model.tokens):,} to target {vocab_size:,}")
    
    # Keep tokens sorted by score (highest first)
    sorted_tokens = sorted(model.tokens, key=lambda t: -t.log_prob)
    
    # Ensure we keep all required base tokens
    base_token_set = {(t,) for t in pretokenizer.base_tokens}
    base_token_ids = {i for i, t in enumerate(model.tokens) if tuple(t.base_tokens) in base_token_set}
    
    # Select top tokens, making sure to keep all base tokens
    selected_tokens = []
    selected_ids = set()
    
    # First add all base tokens
    for token in sorted_tokens:
        if token.id in base_token_ids and token.id not in selected_ids:
            selected_tokens.append(token)
            selected_ids.add(token.id)
    
    # Then add remaining tokens by score until we reach vocab_size
    for token in sorted_tokens:
        if len(selected_tokens) >= vocab_size:
            break
        if token.id not in selected_ids:
            selected_tokens.append(token)
            selected_ids.add(token.id)
    
    # Create new model with selected tokens
    new_model = UnigramModel(pretokenizer, selected_tokens)
    
    logger.info(f" ├─ Kept {len(selected_tokens):,} tokens")
    logger.info(f" └─ Removed {len(model.tokens) - len(selected_tokens):,} tokens")
    
    return new_model, len(model.tokens) - len(selected_tokens)

# --- Main Training Function ---

def train_unigram(
    pretokenizer: BasePretokenizer,
    corpus: PretokenizedCorpus,
    additional_vocab_size: int,
    num_workers: int = 1, # TODO: unused/ignored for now
    verbose: bool = True,
    # unigram specific settings, some experimental
    max_token_len: int = 16,
    initial_vocab_factor: int = 4,
    pre_final_vocab_factor: float = 1.1,
    pruning_shrinking_factor: float = 0.75,
    m_step_dp_smoothing: bool = True,
    m_step_low_count_threshold: float = 0.5,
    defensive_prune: bool = False,
    max_iterations: int = 100,
    num_sub_iterations: int = 2,
) -> UnigramModel:
    logger = create_logger("train_unigram", verbose)
    """Trains a Unigram tokenizer model."""

    # initialize vocab and model
    vocab = make_initial_vocab(logger,corpus, pretokenizer, additional_vocab_size * initial_vocab_factor, max_token_len)
    total_pretokens = sum(freq for _, freq in corpus)
    prune_to_vocab_size = int(len(pretokenizer.base_tokens) + additional_vocab_size * pre_final_vocab_factor)
    final_vocab_size = len(pretokenizer.base_tokens) + additional_vocab_size
    totals_removed = defaultdict(list)
    defended_token_ids = set()

    model = UnigramModel(pretokenizer, vocab)

    # EM Training Loop
    for iter in range(max_iterations):
        # Sub-EM Iterations
        for sub_iter in range(num_sub_iterations):
            expected_count, objective, total_tokens = run_e_step(logger=logger, corpus=corpus, model=model)
            model, m_step_removed = run_m_step(logger=logger, pretokenizer=pretokenizer, model=model, expected_count=expected_count, dp_smoothing=m_step_dp_smoothing, k_expected_frequency_threshold=m_step_low_count_threshold)
            totals_removed["M Step Low Count"].append(m_step_removed)
            avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens
            logger.info(f"🔄 EM Iteration {iter + 1}.{sub_iter + 1}. Model size {len(model.tokens):,}")
            logger.debug(f" ├─ Objective: {objective:.4f}")
            logger.debug(f" ├─ Total tokens: {total_tokens:,d}")
            logger.debug(f" └─ Avg tokens/pretoken: {avg_tokens_per_pretoken:.4f}")

        # Check Stopping Condition
        current_size = len(model.tokens)
        if current_size <= prune_to_vocab_size:
            if verbose: 
                logger.info(f"🎯 Target vocabulary size for EM iterations reached")
                logger.debug(f" ├─ Current: {current_size:,}")
                logger.debug(f" └─ Target:  {prune_to_vocab_size:,}")
            break

        # Pruning Step
        model, num_unused, num_pruned, defended_tokens = prune_tokens(logger=logger, corpus=corpus, pretokenizer=pretokenizer, model=model, desired_vocab_size=prune_to_vocab_size, shrinking_factor=pruning_shrinking_factor, defensive=defensive_prune)
        totals_removed["Prune/Zero Count"].append(num_unused)
        totals_removed["Prune/Loss"].append(num_pruned)
        defended_token_ids.update(t.id for t, _ in defended_tokens)

    # Finalization
    model, finalize_removed = finalize_tokens(logger=logger, pretokenizer=pretokenizer, model=model, vocab_size=final_vocab_size)
    totals_removed["Finalize"].append(finalize_removed)

    total_tokens = sum(len(model.make_lattice(pretoken).viterbi()[0])*freq for pretoken, freq in corpus)
    stats = {
        "total_tokens": total_tokens,
        "tokens/pretoken": total_tokens / total_pretokens,
    }
    if verbose:
        num_defended = len(defended_token_ids)
        defended_in_final = [(t, t.log_prob) for t in model.tokens if t.id in defended_token_ids]
        logger.info(f"🎉 Training completed successfully! Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")
        logger.debug("📊 Token Removal Statistics:")
        for key, value in totals_removed.items():
            logger.debug(f" ├─ {key:<20} {sum(value):6,d} tokens" + (f" in steps {value}" if len(value) > 1 else ""))
        if defensive_prune:
            logger.debug(f" ├─ Defended {num_defended:,} tokens from being removed along with their alternatives.")
            if defended_in_final:
                logger.debug(f" ├─ {len(defended_in_final):,} defended tokens made it to the final vocabulary.")
                log_examples(logger, pretokenizer, defended_in_final, "logprob")
            else:
                logger.debug(" ├─ No defended tokens made it to the final vocabulary.")
        logger.debug("📊 Compression Statistics:")
        logger.debug(f" ├─ Total tokens: {stats['total_tokens']:,d}")
        logger.debug(f" └─ Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")

    model.metadata = stats
    return model
