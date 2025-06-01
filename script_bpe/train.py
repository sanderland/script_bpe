from script_bpe.utils import mp_ctx  # load first. #isort: skip

import argparse
import os
from json import JSONDecodeError

from script_bpe import PRETOKENIZER_REGISTRY, BPETokenizer, get_pretokenizer
from script_bpe.bpe import BPETokenizer, train_bpe
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import PROJECT_ROOT, create_logger

logger = create_logger("main")


# ---- file utils ----


def tokenizer_save_path(file: str, n: int, pretokenizer_name: str) -> str:
    """Get the save path for a tokenizer"""
    return os.path.join(PROJECT_ROOT, f"results/tokenizers/{file}/n{n}/{pretokenizer_name}.json.gz")


def load_tokenizers_for_dataset(file, n):
    """Load all tokenizers for a given dataset and additional vocabulary size."""
    tokenizers = {}
    for ptok in PRETOKENIZER_REGISTRY:
        path = tokenizer_save_path(file, n, ptok)
        tokenizers[ptok] = None
        try:
            tokenizers[ptok] = BPETokenizer.load(path)
            for t in tokenizers[ptok].metadata["tokens"]:
                if t["final_count"] < 0:
                    print(f"Final count for token '{t}' is negative in tokenizer '{ptok}' for {file} at {path}")
        except FileNotFoundError as e:
            pass
        except JSONDecodeError as e:
            print(f"Warning: Tokenizer '{ptok}' at {path} is corrupted or invalid: {e}")
    return tokenizers


# ---- training ----


def train_tokenizer(
    pretokenizer_name,
    corpus_name,
    additional_vocab_size,
    n_cpus: int,
    retrain=False,
    report=False,
) -> BPETokenizer | None:
    save_path = tokenizer_save_path(corpus_name, additional_vocab_size, pretokenizer_name)

    tokenizer = None
    if not retrain:
        try:
            tokenizer = BPETokenizer.load(save_path)
            logger.info(f"Tokenizer loaded from {save_path}")
        except (FileNotFoundError, JSONDecodeError) as e:
            if os.path.exists(save_path):
                logger.error(f"Tokenizer found at {save_path}, but corrupted: {e}")

    if not tokenizer:
        logger.info(
            f"Training tokenizer with {pretokenizer_name} on {corpus_name} with additional vocab size {additional_vocab_size}"
        )
        pretokenizer = get_pretokenizer(pretokenizer_name)
        corpus = load_corpus_by_name(corpus_name, pretokenizer)
        if additional_vocab_size <= 0:  # allows to just prepare corpus
            logger.warning("Additional vocabulary size is 0, skipping training.")
            return None
        tokenizer = train_bpe(
            pretokenizer=pretokenizer,
            corpus=corpus,
            additional_vocab_size=additional_vocab_size,
            num_workers=n_cpus,
        )
        tokenizer.save(save_path)
        logger.info(f"Saved tokenizer to {save_path}")

    if report:
        report_path = save_path.replace(".json.gz", ".md")
        with open(report_path, "w") as report_file:
            report_file.write(tokenizer.report())
        logger.info(f"Saved report to {report_path}")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE with optional profiling.")
    parser.add_argument(
        "--dataset", type=str, default="catherinearnett/monolingual-tokenizer-data", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--corpus", type=str, default="kor_hang_300mb", help="Name for corpus to train on, see load_dataset"
    )
    parser.add_argument("--pretokenizer", type=str, default="scriptenc_cb", help="Pretokenizer name")
    parser.add_argument("--additional_vocab_size", "-n", type=int, default=100, help="Additional vocabulary size")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of the tokenizer, even if it exists")
    parser.add_argument("--parallel", "-p", type=int, default=4, help="Number of CPUs to use for training")
    parser.add_argument("--report", action="store_true", help="Generate a report after training")
    args = parser.parse_args()

    train_tokenizer(
        pretokenizer_name=args.pretokenizer,
        corpus_name=args.corpus,
        additional_vocab_size=args.additional_vocab_size,
        retrain=args.retrain,
        n_cpus=args.parallel,
        report=args.report,
    )


if __name__ == "__main__":
    main()
