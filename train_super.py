#!/usr/bin/env python3

from collections import Counter
from typing import Iterator
import os

from script_bpe.bpe import BPETokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.train import tokenizer_save_path
import tqdm


def generate_ngrams(tokens: list[int], n: int) -> Iterator[tuple[int, ...]]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def count_ngrams(
    corpus_name: str,
    pretokenizer_name: str,
    vocab_size: int = 64000,
    max_docs: int | None = None,
    max_n: int = 4
) -> dict[int, Counter]:
    """
    Load dataset as raw texts, tokenize using a trained tokenizer, and count unique n-grams.
    
    Args:
        corpus_name: Name of the corpus to load
        pretokenizer_name: Name of the pretokenizer to use
        vocab_size: Vocabulary size of the trained tokenizer to load
        max_docs: Maximum number of documents to process (None for all)
        max_n: Maximum n-gram size (default 4, range will be 2 to max_n inclusive)
    
    Returns:
        Dictionary mapping n-gram size (2 to max_n) to Counter of n-grams
    """
    print(f"Loading trained tokenizer: {pretokenizer_name} with vocab size {vocab_size}")
    save_path = tokenizer_save_path(corpus_name, vocab_size, pretokenizer_name)
    try:
        tokenizer = BPETokenizer.load(save_path)
        print(f"Loaded tokenizer from: {save_path}")
    except FileNotFoundError:
        print(f"Tokenizer not found at {save_path}")
        print("You may need to train the tokenizer first using script_bpe/train.py")
        raise

    print(f"Loading corpus as raw texts: {corpus_name}")
    # Load corpus as raw texts instead of pre-tokenized chunks
    ds = load_corpus_by_name(corpus_name, None, return_dataset=True)
    
  
    # Counters for each n-gram size
    ngram_counters = {
        n: Counter() for n in range(2, max_n + 1)
    }
    
    for doc_idx, row in tqdm.tqdm(enumerate(ds)):
        text = row["text"]
        if max_docs is not None and doc_idx >= max_docs:
            break
            
        # Encode the entire text with keep_chunks=True
        enc_chunks = tokenizer.encode(text, keep_chunks=True)
        # Generate n-grams from the chunks
        for n in range(2, max_n + 1):
            for i in range(len(enc_chunks) - n + 1):
                ngram = enc_chunks[i:i+n]
                # Only count n-grams where each chunk is a single token
                if all(len(chunk) == 1 for chunk in ngram):
                    flat_ngram = tuple(chunk[0] for chunk in ngram)
                    ngram_counters[n][flat_ngram] += 1
    
    for n in range(2, max_n + 1):
        print(f"Unique {n}-grams: {len(ngram_counters[n]):,}")
        print(f"Total {n}-gram occurrences: {sum(ngram_counters[n].values()):,}")
    
    return ngram_counters


def print_top_ngrams(ngram_counters: dict[int, Counter], tokenizer: BPETokenizer, top_k: int = 100, max_n: int = 4):
    """Print the most frequent n-grams."""
    
    for n in range(2, max_n + 1):
        print(f"\nTop {top_k} {n}-grams:")
        print("-" * 50)
        
        for i, (ngram, count) in enumerate(ngram_counters[n].most_common(top_k), 1):
            # Decode the n-gram tokens back to readable text
            readable_tokens = []
            for token_id in ngram:
                if token_id in tokenizer.tokens:
                    # Try to decode the token to readable text
                    try:
                        token_text = tokenizer.decode([token_id])
                        readable_tokens.append(repr(token_text))
                    except:
                        readable_tokens.append(f"<{token_id}>")
                else:
                    readable_tokens.append(f"<{token_id}>")
            
            tokens_str = " + ".join(readable_tokens)
            print(f"{i:2d}. {tokens_str:<60} (count: {count:,})")


def main():
    """Main function to run the n-gram counting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Count n-grams in a dataset")
    parser.add_argument("-c", "--corpus", default="eng_latn_300mb", 
                       help="Corpus name to load")
    parser.add_argument("-p", "--pretokenizer", default="scriptenc_cb",
                       help="Pretokenizer to use")
    parser.add_argument("-n", "--vocab-size", type=int, default=64000,
                       help="Vocabulary size of the trained tokenizer")
    parser.add_argument("-d", "--max-docs", type=int, default=None,
                       help="Maximum number of documents to process")
    parser.add_argument("-k", "--top-k", type=int, default=20,
                       help="Number of top n-grams to display")
    parser.add_argument("--max-n", type=int, default=4,
                       help="Maximum n-gram size (range will be 2 to max-n inclusive)")
    parser.add_argument("-s", "--save-results", action="store_true",
                       help="Save results to JSON file")
    args = parser.parse_args()
    
    # Count n-grams
    ngram_counters = count_ngrams(
        corpus_name=args.corpus,
        pretokenizer_name=args.pretokenizer,
        vocab_size=args.vocab_size,
        max_docs=args.max_docs,
        max_n=args.max_n
    )
    
    # Load tokenizer for printing results
    save_path = tokenizer_save_path(args.corpus, args.vocab_size, args.pretokenizer)
    tokenizer = BPETokenizer.load(save_path)
    
    # Print top n-grams
    print_top_ngrams(ngram_counters, tokenizer, args.top_k, args.max_n)
    
    # Save results if requested
    if args.save_results:
        import json
        
        # Convert Counter objects to regular dicts for JSON serialization
        # Only save the counts, not the full n-gram tuples (too large)
        results = {
            "metadata": {
                "corpus": args.corpus,
                "pretokenizer": args.pretokenizer,
                "vocab_size": args.vocab_size,
                "max_docs": args.max_docs,
                "max_n": args.max_n,
            },
            "summary": {
                f"unique_{n}grams": len(ngram_counters[n])
                for n in range(2, args.max_n + 1)
            },
            "summary_totals": {
                f"total_{n}gram_occurrences": sum(ngram_counters[n].values())
                for n in range(2, args.max_n + 1)
            },
            "top_ngrams": {
                str(n): [
                    {"ngram": list(ngram), "count": count}
                    for ngram, count in ngram_counters[n].most_common(args.top_k)
                ]
                for n in range(2, args.max_n + 1)
            }
        }
        
        filename = f"results/super/ngram_results_{args.corpus}_{args.pretokenizer}_v{args.vocab_size}_n{args.max_n}.json"
        if args.max_docs is not None:
            filename = filename.replace(".json", f"_test.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
