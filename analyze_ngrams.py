#%%
import os
import json
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from script_bpe.bpe.fast.tokenizer import FastScriptTokenizer
from paper_utils.benchmark_cpp import TOKENIZER_BASE_PATH

# Settings for each dataset
DATASETS = [
    {
        'name': 'CulturaX-subsample-100-bal2',
        'hf_name': 'sanderland/CulturaX-subsample-100-bal2',
        'vocab_size': 256000,
        'tokenizer_name': 'CulturaX-subsample-100-bal2',
        'tokenizer_file': 'scriptenc_cb.json.gz',
        'output': 'ngrams/cultura-ngrams.json',
    },
    {
        'name': 'lmsys-chat-1m',
        'hf_name': 'lmsys/lmsys-chat-1m',
        'vocab_size': 256000,  # Use same vocab size for consistency
        'tokenizer_name': 'CulturaX-subsample-100-bal2',
        'tokenizer_file': 'scriptenc_cb.json.gz',
        'output': 'ngrams/lmsys-ngrams.json',
    },
]
DATASETS = DATASETS[1:]
print(f"Processing {len(DATASETS)} datasets: {[d['name'] for d in DATASETS]}")

NGRAMS = [1, 2, 3]
CHUNK_SIZE = 1_000_000  # Number of docs per chunk
os.makedirs('cache', exist_ok=True)

def get_tokenizer(corpus_name, vocab_size, tokenizer_file):
    tokenizer_path = os.path.join(
        TOKENIZER_BASE_PATH, corpus_name, f"n{vocab_size}", tokenizer_file
    )
    print(f"Loading tokenizer from {tokenizer_path}")
    return FastScriptTokenizer.load(tokenizer_path)

def iter_texts(row):
    if 'text' in row:
        yield row['text']
    if 'conversation' in row:
        content = "\n".join(turn['content'] for turn in row['conversation'])
        yield content

def count_ngrams(tokens, ngrams, total):
    for n in ngrams:
        totals_n = total[n]
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            totals_n[ng] += 1

def merge_counts(total, part):
    for n in total:
        total[n].update(part[n])

def process_dataset(cfg):
    print(f"Processing {cfg['name']}")
    tokenizer = get_tokenizer(cfg['tokenizer_name'], cfg['vocab_size'], cfg['tokenizer_file'])
    dataset = load_dataset(cfg['hf_name'], split='train', streaming=True)
    ngram_counts = {n: Counter() for n in NGRAMS}
    total = 0
    for idx, row in tqdm(enumerate(dataset), desc=cfg['name']):
        for text in iter_texts(row):
            tokens = tokenizer.encode(str(text))
            total += 1
            count_ngrams(tokens.tolist(), NGRAMS, ngram_counts)
        if (idx + 1) % 250000 == 0:
            print(f"Processed {idx + 1} documents, ngram sizes:")
            for n in NGRAMS:
                print(f"  {n}-grams: {len(ngram_counts[n])} unique")
        if (idx + 1) % CHUNK_SIZE == 0:
            for i in [2,3]:
                ngram_counts[i] = Counter({k: v for k, v in ngram_counts[i].items() if v > 1})
                print(f"Chunk {idx + 1} processed, {i}-grams reduced to {len(ngram_counts[i])} unique")
    # Save
    out = {str(n): {" ".join(map(str, k)): v for k, v in ngram_counts[n].items()} for n in NGRAMS}
    os.makedirs(os.path.dirname(cfg['output']), exist_ok=True)
    print(f"Total documents processed: {total}")
    print(f"Total n-grams counted: {sum(len(ngram_counts[n]) for n in NGRAMS)}")
    with open(cfg['output'], 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved n-gram counts for {cfg['name']} to {cfg['output']}")

if __name__ == "__main__":
    for cfg in DATASETS:
        process_dataset(cfg)
