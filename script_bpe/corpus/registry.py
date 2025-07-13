import os

from datasets import Dataset, load_dataset

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.utils import create_logger


def create_huggingface_corpus(
    dataset_name: str, corpus_name: str, pretokenizer, base_dir: str, logger, return_dataset: bool = False, **kwargs
) -> PretokenizedCorpus | Dataset:
    num_cpus = os.cpu_count() or 4
    dataset = load_dataset(dataset_name, **kwargs)
    if return_dataset:
        return dataset
    logger.info(f"Loaded dataset {dataset_name} with args {kwargs}, pretokenizing on {num_cpus} CPUs.")
    corpus = PretokenizedCorpus.from_texts(
        name=corpus_name,
        base_path=base_dir,
        pretokenizer=pretokenizer,
        texts=[doc["text"] for doc in dataset],  # Assuming the dataset has a "text" field
        num_workers=num_cpus,
    )
    logger.info(f"Created corpus {corpus_name} with pretokenizer {pretokenizer.hash()} in {corpus.dir_path()}")
    return corpus


def load_corpus_by_name(
    corpus_name,
    pretokenizer,
    base_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH,
    return_dataset: bool = False,
) -> PretokenizedCorpus | Dataset:  # little hardcoded dataset registry
    logger = create_logger("corpus")
    if not return_dataset:
        try:
            corpus = PretokenizedCorpus(
                name=corpus_name,
                base_path=base_dir,
                pretokenizer=pretokenizer,
            )
            return corpus
        except FileNotFoundError as e:
            logger.warning(
                f"Corpus {corpus_name} with pretokenizer {pretokenizer.hash()} not found in cache, creating it: {e}"
            )

    if corpus_name.endswith("300mb"):
        return create_huggingface_corpus(
            "catherinearnett/monolingual-tokenizer-data",
            corpus_name=corpus_name,
            base_dir=base_dir,
            pretokenizer=pretokenizer,
            logger=logger,
            split="train",
            data_files=[f"{corpus_name}.txt"],
            return_dataset=return_dataset,
        )
    elif "OSCAR" in corpus_name or "CulturaX" in corpus_name:
        return create_huggingface_corpus(
            f"sanderland/{corpus_name}",
            corpus_name=corpus_name,
            base_dir=base_dir,
            logger=logger,
            pretokenizer=pretokenizer,
            split="train",
            return_dataset=return_dataset,
        )
    elif corpus_name == "swift":
        with open("tests/data/taylorswift.txt", "r") as f:
            if return_dataset:
                return [f.read()]
            return PretokenizedCorpus.from_texts(corpus_name, pretokenizer=pretokenizer, texts=[f.read()])
    else:
        raise ValueError(f"Unknown dataset: {corpus_name}")


MONOLINGUAL_DATASETS = [  # in order of average number of bytes/char in dataset
    "eng_latn_300mb",  # ~1B/char: ASCII only, very efficient
    "deu_latn_300mb",  # ~1B/char: mostly ASCII, occasional umlauts (2B)
    "vie_latn_300mb",  # ~1.3B/char: Latin with many combining diacritics (2–3B per char possible)
    "heb_hebr_300mb",  # ~2B/char: Hebrew uses 2-byte characters in UTF-8
    "arb_arab_300mb",  # ~2/char: Arabic base chars = 2B; some diacritics & shaping
    "rus_cyrl_300mb",  # ~2B/char: Cyrillic mostly 2-byte characters
    "kor_hang_300mb",  # ~3B/char: Hangul syllables are full 3B in UTF-8
    "hin_deva_300mb",  # ~3B/char: Devanagari has many combining marks, 3B typical
    "tha_thai_300mb",  # ~3B/char: Thai script, including tone marks and vowels
    "zho_hans_300mb",  # ~3B/char: Simplified Chinese, each Han character is 3B
    "jpn_jpan_300mb",  # ~3B/char: Mix of of 3B scripts
    "pan_guru_300mb",  # ~3B/char: Gurmukhi script, mostly 3B characters
]
