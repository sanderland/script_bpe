#%%
from paper_utils.benchmark_cpp import TOKENIZER_BASE_PATH
from script_bpe.bpe.tokenizer import BPETokenizer
import os
from script_bpe.bpe.fast.tokenizer import FastScriptTokenizer

tokenizer_file = 'scriptenc_cb.json.gz'
vocab_size = 64000


corpus_name = 'deu_latn_300mb'
text ='seit langem ̶ durchaus offensiv'

text = "ます🙇\u200d♀️"
corpus_name = 'jpn_jpan_300mb'

text = "\n1  koppie farro\n2 koppies grondboontjie"
corpus_name = "CulturaX-subsample-100-bal2"
vocab_size = 256000

text = "이에요 ̈ 저희"
corpus_name = 'kor_hang_300mb'
vocab_size = 64000


tokenizer_path = os.path.join(
    TOKENIZER_BASE_PATH, corpus_name, f"n{vocab_size}", tokenizer_file
)
print(f"Tokenizing with {tokenizer_path}")
tokenizer = BPETokenizer.load(tokenizer_path)
pretokenizer = tokenizer.pretokenizer

fast_tokenizer = FastScriptTokenizer.load(tokenizer_path)




print(f"Text: {text!r}")
chunks = pretokenizer.encode_and_chunk(text)
for c in chunks:
    print(f"{c} {pretokenizer.decode(c)!r}")
    for i in range(0, len(c), 2):
        pair = (c[i], c[i+1])
        ch = pretokenizer.decode(pair)
        print(f"  {i//2}: {pair} {ch!r} U+{ord(ch):04X} {pretokenizer.tokens_to_readable_string(pair[:1])!r} {pretokenizer.tokens_to_readable_string(pair[1:])!r}")

tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)!r}")
for t in tokens:
    print(f"  {t} {tokenizer.decode([t], errors='backslashreplace')!r}")

print(f"\n--- FastBPETokenizer (C++ backend) ---")
fast_tokens = fast_tokenizer.encode(text)
print(f"Tokens: {fast_tokens}")
print(f"Decoded: {fast_tokenizer.decode(fast_tokens)!r}")
for t in fast_tokens:
    print(f"  {t} {fast_tokenizer.decode([t], errors='backslashreplace')!r}")




