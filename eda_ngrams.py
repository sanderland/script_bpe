#%%
import os
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

# --- Configuration ---

# This base path should point to the directory containing your tokenizer files
# Example structure: ./tokenizers/CulturaX-subsample-100-bal2/n256000/scriptenc_cb.json.gz
TOKENIZER_BASE_PATH = Path("./tokenizers") 
# This path should point to the directory containing your ngram count files
NGRAM_BASE_PATH = Path("./ngrams")

# Define the two corpora to compare
# Corpus 1: The baseline/pre-training corpus
corpus1_config = {
    "name": "CulturaX",
    "corpus_name_fs": "CulturaX-subsample-100-bal2",
    "tokenizer_file": "scriptenc_cb.json.gz",
    "ngram_file": "culturax-ngrams.json",
    "vocab_size": 256000,
}

# Corpus 2: The comparison/post-training corpus
corpus2_config = {
    "name": "LMSys-Chat",
    "corpus_name_fs": "lmsys-chat-1m",
    "tokenizer_file": "scriptenc_cb.json.gz",
    "ngram_file": "lmsys-ngrams.json",
    "vocab_size": 256000,
}

# How much less frequent must an ngram be in Corpus 2 to be highlighted?
# e.g., 0.1 means it's highlighted if its frequency is <10% of its frequency in Corpus 1.
DISPARITY_RATIO_THRESHOLD = 0.1
# How many top results to show for each n-gram size
TOP_N_RESULTS = 25


# --- Data Loading and Processing ---

def load_corpus_data(config):
    """Loads tokenizer and ngram data, and calculates total token count."""
    print(f"--- Loading data for corpus: {config['name']} ---")
    
    # Load n-gram results
    ngram_json_path = NGRAM_BASE_PATH / config['ngram_file']
    print(f"Loading n-grams from: {ngram_json_path}")
    with open(ngram_json_path, 'r') as f:
        ngram_data = json.load(f)

    # Calculate total token count (sum of all unigram counts) for normalization
    total_unigram_count = sum(entry['count'] for entry in ngram_data['top_ngrams']['1'])
    print(f"Total unigrams (tokens): {total_unigram_count:,}")

    # Create a lookup for fast access: {'ngram_str': {'count': X, 'freq': Y}}
    ngram_lookup = {}
    for n_str, entries in ngram_data['top_ngrams'].items():
        for entry in entries:
            # Use a consistent tuple string representation for keys
            ngram_key = str(tuple(entry['ngram']))
            ngram_lookup[ngram_key] = {
                "count": entry['count'],
                "freq": entry['count'] / total_unigram_count
            }
            
    return {"name": config['name'], "lookup": ngram_lookup, "total_tokens": total_unigram_count}

# Load data for both corpora
corpus1_data = load_corpus_data(corpus1_config)
corpus2_data = load_corpus_data(corpus2_config)

# --- Analysis ---

# Set the frequency threshold based on the least frequent unigram in the first corpus.
# This gives us a meaningful floor for what we consider "significant".
min_unigram_freq_c1 = min(
    item['freq'] for key, item in corpus1_data['lookup'].items() if key.count(',') == 0
)
print(f"\nMinimum unigram frequency in {corpus1_data['name']}: {min_unigram_freq_c1:.2e}")
print(f"This will be the threshold for displaying n-grams.")
print(f"An n-gram will be highlighted in orange if its frequency in {corpus2_data['name']} is less than {DISPARITY_RATIO_THRESHOLD:.0%} of its frequency in {corpus1_data['name']}.\n")


console = Console()

for n in [2, 3, 4]:
    
    # --- Collect and Compare N-grams ---
    comparison_results = []
    
    # Find all n-grams in Corpus 1 that are above our frequency threshold
    n_str_to_check = str(n)
    
    # Filter keys from the lookup that correspond to the current n-gram size
    corpus1_ngrams = {
        key: val for key, val in corpus1_data['lookup'].items() 
        if key.count(',') == n - 1
    }

    for ngram_key, c1_stats in corpus1_ngrams.items():
        # Apply the frequency threshold
        if c1_stats['freq'] < min_unigram_freq_c1:
            continue
            
        # Get stats for the same ngram from Corpus 2, default to 0 if not present
        c2_stats = corpus2_data['lookup'].get(ngram_key, {"count": 0, "freq": 0.0})
        
        comparison_results.append({
            "ngram": ngram_key,
            "freq1": c1_stats['freq'],
            "freq2": c2_stats['freq'],
        })

    # Sort by frequency in Corpus 1 (descending)
    comparison_results.sort(key=lambda x: x['freq1'], reverse=True)


    # --- Display Results in a Table ---
    
    table = Table(
        title=f"Top {n}-grams More Common in '{corpus1_data['name']}' than '{corpus2_data['name']}' (showing top {TOP_N_RESULTS})",
        show_lines=True
    )
    table.add_column("N-gram", justify="left", style="cyan", no_wrap=True)
    table.add_column(f"Fraction ({corpus1_data['name']})", justify="right", style="magenta")
    table.add_column(f"Fraction ({corpus2_data['name']})", justify="right", style="green")
    table.add_column("Ratio (C1/C2)", justify="right", style="yellow")
    
    for result in comparison_results[:TOP_N_RESULTS]:
        freq1 = result['freq1']
        freq2 = result['freq2']
        
        # Avoid division by zero
        ratio = freq1 / freq2 if freq2 > 0 else float('inf')
        ratio_str = f"{ratio:,.1f}x" if freq2 > 0 else "∞"
        
        # Determine row style (highlight if disparity is large)
        row_style = ""
        if freq2 < freq1 * DISPARITY_RATIO_THRESHOLD:
            row_style = "bright_red" # Changed to red for better visibility

        table.add_row(
            result['ngram'],
            f"{freq1:.3e}",
            f"{freq2:.3e}",
            ratio_str,
            style=row_style
        )
        
    console.print(table)


#%%