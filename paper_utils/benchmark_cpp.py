import os
import time
import argparse
from typing import List
from datasets import load_dataset
from tabulate import tabulate

# Add the project root to the Python path to allow imports from script_bpe
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from script_bpe.bpe import BPETokenizer
from script_bpe.bpe.fast.tokenizer import FastScriptTokenizer

# List of monolingual corpora based on the directory structure provided
MONOLINGUAL_CORPORA = [
    "eng_latn_300mb",
    "deu_latn_300mb",
    "vie_latn_300mb",
    "rus_cyrl_300mb",
    "heb_hebr_300mb",
    "arb_arab_300mb",
    "hin_deva_300mb",
    "pan_guru_300mb",
    "tha_thai_300mb",
    "kor_hang_300mb",
    "jpn_jpan_300mb",
    "zho_hans_300mb",
]

TOKENIZER_BASE_PATH = "results/tokenizers"

def time_encoding(tokenizer, texts: List[str]) -> (float, int):
    """
    Measures the time taken to encode a list of texts and returns the duration
    and total number of tokens produced.
    """
    start_time = time.perf_counter()
    total_tokens = 0
    for text in texts:
        # The actual work being measured
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return duration, total_tokens

def main(args):
    """Main function to run the benchmark."""
    results = []
    print(f"--- Tokenizer Speed Benchmark ---")
    print(f"Comparing Python vs. C++ on {len(MONOLINGUAL_CORPORA)} corpora.")
    print(f"Using first {args.n_docs} documents from each corpus.")
    print(f"Tokenizer: {args.tokenizer_name} (n={args.vocab_size})\n")

    for corpus_name in MONOLINGUAL_CORPORA:
        print(f"Processing '{corpus_name}'...")

        # 1. Load the pre-trained Python tokenizer
        tokenizer_file = args.tokenizer_name
        if not tokenizer_file.endswith(".json.gz"):
            tokenizer_file += ".json.gz"
            
        tokenizer_path = os.path.join(
            TOKENIZER_BASE_PATH, corpus_name, f"n{args.vocab_size}", tokenizer_file
        )
        if not os.path.exists(tokenizer_path):
            print(f"  -> SKIPPING: Tokenizer not found at {tokenizer_path}")
            continue

        try:
            python_tokenizer = BPETokenizer.load(tokenizer_path)
        except Exception as e:
            print(f"  -> SKIPPING: Failed to load tokenizer {tokenizer_path}. Error: {e}")
            continue

        # 2. Instantiate the fast C++ tokenizer from the loaded Python one
        try:
            cpp_tokenizer = FastScriptTokenizer(
                merge_rules=python_tokenizer.merge_rules,
                pretokenizer=python_tokenizer.pretokenizer,
                metadata=python_tokenizer.metadata
            )
        except RuntimeError as e:
            print(f"  -> SKIPPING: Could not instantiate FastScriptTokenizer for {corpus_name}. Is it 'scriptenc_cb'?\n     Error: {e}")
            continue

        # 3. Load the test data efficiently
        print("  -> Loading dataset...")
        try:
            dataset = load_dataset(
                "catherinearnett/monolingual-tokenizer-data",
                data_files=[f"{corpus_name}.txt"],
                split="train",
                streaming=True,
            )
            test_texts = [doc['text'] for doc in dataset.take(args.n_docs)]
        except Exception as e:
            print(f"  -> SKIPPING: Failed to load dataset for {corpus_name}. Error: {e}")
            continue
            
        if not test_texts:
            print(f"  -> SKIPPING: No text data loaded for {corpus_name}.")
            continue

        # 4. Run benchmarks
        print("  -> Benchmarking Python implementation...")
        py_time, total_tokens = time_encoding(python_tokenizer, test_texts)
        
        print("  -> Benchmarking C++ implementation...")
        cpp_time, _ = time_encoding(cpp_tokenizer, test_texts)

        # 5. Calculate and store results
        speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
        tokens_per_sec_py = total_tokens / py_time if py_time > 0 else 0
        tokens_per_sec_cpp = total_tokens / cpp_time if cpp_time > 0 else 0
        
        results.append({
            "Corpus": corpus_name.replace("_300mb", ""),
            "Python (s)": f"{py_time:.3f}",
            "C++ (s)": f"{cpp_time:.3f}",
            "Speedup": f"{speedup:.2f}x",
            "Py Tok/s": f"{tokens_per_sec_py:,.0f}",
            "C++ Tok/s": f"{tokens_per_sec_cpp:,.0f}",
        })
        print(f"  -> Done. Speedup: {speedup:.2f}x\n")

    # 6. Print final summary table
    if results:
        print("\n--- Benchmark Summary ---")
        print(tabulate(results, headers="keys", tablefmt="github"))
    else:
        print("\nNo benchmarks were successfully run. Please check paths and configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs. C++ tokenizer implementations."
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=10000,
        help="Number of documents from each corpus to use for testing.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=64000,
        help="The vocabulary size 'n' used in the tokenizer path (e.g., n64000).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="scriptenc_cb",
        help="The name of the tokenizer file to test (e.g., 'scriptenc_cb').",
    )
    args = parser.parse_args()
    main(args)