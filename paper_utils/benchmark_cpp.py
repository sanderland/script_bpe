import os
import time
import argparse
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
#MONOLINGUAL_CORPORA = ["deu_latn_300mb"]

def time_encoding(tokenizer, texts: list[str]) -> tuple[float, int, list[list[int]]]:
    """
    Measures the time taken to encode a list of texts and returns the duration
    and total number of tokens produced.
    """
    start_time = time.perf_counter()
    total_tokens = 0
    tokenized = []
    for text in texts:
        # The actual work being measured
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        tokenized.append(encoded)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return duration, total_tokens, tokenized

def main(args):
    """Main function to run the benchmark."""
    results = []
    print(f"--- Tokenizer Speed Benchmark ---")
    print(f"Comparing Python vs. C++ on {len(MONOLINGUAL_CORPORA)} corpora.")
    print(f"Using first {args.n_docs} documents from each corpus.")
    print(f"Tokenizer: {args.tokenizer_name} (n={args.vocab_size})\n")

    for corpus_name in MONOLINGUAL_CORPORA:
        # Consistent width for emoji and log text
        log_width = 32
        def elog(emoji, msg):
            return f"  {emoji:<3} {msg:<{log_width}}"

        print(f"\n{elog('🔍', f"Processing '{corpus_name}'...")}")

        # 1. Load the tokenizers
        tokenizer_file = args.tokenizer_name
        if not tokenizer_file.endswith(".json.gz"):
            tokenizer_file += ".json.gz"

        tokenizer_path = os.path.join(
            TOKENIZER_BASE_PATH, corpus_name, f"n{args.vocab_size}", tokenizer_file
        )
        if not os.path.exists(tokenizer_path):
            print(elog('⚠️', f"SKIPPING: Tokenizer not found at {tokenizer_path}"))
            continue

        python_tokenizer = BPETokenizer.load(tokenizer_path)
        cpp_tokenizer = FastScriptTokenizer(
            merge_rules=python_tokenizer.merge_rules,
            pretokenizer=python_tokenizer.pretokenizer,
            metadata=python_tokenizer.metadata
        )

        # 2. Load the test data efficiently
        print(elog('💾', "Loading dataset..."))
        dataset = load_dataset(
            "catherinearnett/monolingual-tokenizer-data",
            data_files=[f"{corpus_name}.txt"],
            split="train",
            streaming=True,
        )
        test_texts = [doc['text'] for doc in dataset.take(args.n_docs)]

        # 3. Run benchmarks
        print(elog('🐍', "Benchmarking Python implementation..."))
        py_time, total_tokens, py_tokenized = time_encoding(python_tokenizer, test_texts)

        print(elog('🅲➕➕',"Benchmarking C++ implementation... 🚀"))
        cpp_time, _, cpp_tokenized = time_encoding(cpp_tokenizer, test_texts)

        n_mismatches = 0
        if any(len(py_tok_i)!=len(cpp_tok_i) or (py_tok_i != cpp_tok_i).any() for py_tok_i, cpp_tok_i in zip(py_tokenized, cpp_tokenized)):
            print(elog('❌', "Tokenization mismatch"))
            for i in range(len(py_tokenized)):
                if (py_tokenized[i] != cpp_tokenized[i]).any():
                    n_mismatches += 1
                    print(f"    {'❌':<2} Mismatch at index {i}")
                    print(f"      Python: {py_tokenized[i]}")
                    print(f"      C++   : {cpp_tokenized[i]}")
                    print(f"      Text  : {test_texts[i]!r}")
                    py_decoded = python_tokenizer.decode(py_tokenized[i])
                    cpp_decoded = cpp_tokenizer.decode(cpp_tokenized[i])
                    print(f"      Python decoded: {py_decoded!r}")
                    if py_decoded != cpp_decoded:
                        print(f"      C++ decoded   : {cpp_decoded!r}")
                    else:
                        print(f"      C++ decoded   Matches Python")
                    for ti, (tpy, tcpp) in enumerate(zip(py_tokenized[i], cpp_tokenized[i])):
                        mm =  "❌" if tpy != tcpp else "✅"
                        print(f"      {mm} {ti}: py {tpy}  {python_tokenizer.decode([tpy])!r} cpp {tcpp} {cpp_tokenizer.decode([tcpp])!r} {mm}")
                        if tpy!=tcpp:
                            break

        # 4. Calculate and store results
        speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
        tokens_per_sec_py = total_tokens / py_time if py_time > 0 else 0
        tokens_per_sec_cpp = total_tokens / cpp_time if cpp_time > 0 else 0

        matches = "✅" if n_mismatches == 0 else f"❌ {n_mismatches}"
        results.append({
            "Corpus": corpus_name.replace("_300mb", ""),
            "Python (s)": f"{py_time:.3f}",
            "C++ (s)": f"{cpp_time:.3f}",
            "Py Tok/s": f"{tokens_per_sec_py:,.0f}",
            "C++ Tok/s": f"{tokens_per_sec_cpp:,.0f}",
            "Matches": matches,
            "Speedup": f"{speedup:.2f}x",
        })
        print(elog('✅' if n_mismatches == 0 else '❌', f"Done. Speedup: {speedup:.2f}x") + "\n")

    # 6. Print final summary table with total time and mean speedup
    if results:
        print("\n--- Benchmark Summary ---")
        print(tabulate(results, headers="keys", tablefmt="github"))

        # Calculate total time and mean speedup
        total_py = sum(float(r["Python (s)"]) for r in results)
        total_cpp = sum(float(r["C++ (s)"]) for r in results)
        mean_speedup = sum(float(r["Speedup"].replace('x','')) for r in results) / len(results)
        print("\n--- Totals ---")
        print(f"{'Total Python time:':<22} {total_py:.2f} s")
        print(f"{'Total C++ time:':<22} {total_cpp:.2f} s")
        print(f"{'Mean speedup:':<22} {mean_speedup:.2f}x")
    else:
        print("\nNo benchmarks were successfully run. Please check paths and configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs. C++ tokenizer implementations."
    )
    parser.add_argument(
        "--n-docs",
        "-n",
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