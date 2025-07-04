"""Tests for FastScriptTokenizer C++ implementation."""

import pytest
import tempfile
import shutil
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.bpe.train import train_bpe
from script_bpe.bpe.fast.tokenizer import FastScriptTokenizer
from script_bpe.corpus import PretokenizedCorpus


@pytest.fixture(scope="module")
def temp_dir_module():
    """Create a temporary directory for the module for efficiency."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def bug_trigger_tokenizer(temp_dir_module):
    """
    Creates a tokenizer with a very specific, skewed corpus. This is designed to create a
    reproducible set of pre-merged and unmerged characters to validate the core C++
    processing logic and prevent regressions. It ensures:
    - 'a' is pre-merged into a single character token.
    - 'c' remains an unmerged base pair.
    - A merge rule for the token of 'a' and the base token of 'c' exists.
    """
    pretokenizer = get_pretokenizer("scriptenc_cb")
    corpus_text = ("ac" * 200) + ("a" * 100) + "c"
    corpus = PretokenizedCorpus.from_texts(
        name="test_corpus_bug_trigger",
        texts=[corpus_text], pretokenizer=pretokenizer, base_path=temp_dir_module
    )
    tokenizer = train_bpe(
        pretokenizer, corpus, additional_vocab_size=10, num_workers=1, verbose=False
    )
    return tokenizer, pretokenizer


@pytest.fixture(scope="module")
def multilingual_tokenizer(temp_dir_module):
    """
    Creates a tokenizer trained on a diverse, multilingual corpus to ensure broad
    compatibility and test handling of various scripts.
    """
    pretokenizer = get_pretokenizer("scriptenc_cb")
    corpus = PretokenizedCorpus.from_texts(
        name="test_corpus_multilingual",
        texts=[
            "Hello world! This is a test.",
            "Script encoding works: 世界你好, and continues.",
            "Testing with äöü ñ ç and other characters.",
            "Numbers: 123 456!@#$%",
            "Mixed scripts: Здравствуй мир! שלום עולם!",
            "Emojis are fun 🌍🌎🌏 and symbols too ©®™.",
        ] * 10, # Repeat to get some merges
        pretokenizer=pretokenizer,
        base_path=temp_dir_module
    )
    tokenizer = train_bpe(
        pretokenizer, corpus, additional_vocab_size=100, num_workers=1, verbose=False
    )
    return tokenizer, pretokenizer


# Test cases for the bug-triggering tokenizer
BUG_TRIGGER_CASES = [
    pytest.param("a", "Single pre-merged char", id="bug_single_merged"),
    pytest.param("c", "Single unmerged char", id="bug_single_unmerged"),
    pytest.param("ac", "Merged followed by unmerged (the critical case)", id="bug_merged_unmerged"),
    pytest.param("ca", "Unmerged followed by merged", id="bug_unmerged_merged"),
    pytest.param("aca", "Unmerged between two merged", id="bug_unmerged_between_merged"),
    pytest.param("cac", "Merged between two unmerged", id="bug_merged_between_unmerged"),
]


@pytest.mark.parametrize("test_text, description", BUG_TRIGGER_CASES)
def test_fast_tokenizer_core_logic(bug_trigger_tokenizer, test_text, description):
    """
    Tests the core C++ processing logic using a specially crafted tokenizer
    to ensure merged/unmerged character sequences are handled correctly.
    """
    python_tok, pretokenizer = bug_trigger_tokenizer
    cpp_tok = FastScriptTokenizer(
        merge_rules=python_tok.merge_rules, pretokenizer=pretokenizer, metadata=python_tok.metadata
    )

    python_tokens = python_tok.encode(test_text)
    cpp_tokens = cpp_tok.encode(test_text)

    assert cpp_tokens.tolist() == python_tokens.tolist(), (
        f"Core logic mismatch for case: '{description}'\n"
        f"String: {repr(test_text)}\n"
        f"Python (correct): {python_tokens.tolist()}\n"
        f"C++    (buggy):   {cpp_tokens.tolist()}"
    )


# Test cases for the multilingual tokenizer
MULTILINGUAL_CASES = [
    pytest.param("", "Empty string", id="multi_empty_string"),
    pytest.param("Hello world!", "Basic Latin", id="multi_latin"),
    pytest.param("Hello, 世界你好", "Mixed Latin and Han", id="multi_latin_han"),
    pytest.param("Здравствуй-мир", "Cyrillic with punctuation", id="multi_cyrillic_punct"),
    pytest.param("123שלום 456", "Hebrew with numbers and space", id="multi_hebrew_nums"),
    pytest.param("🌍🌎🌏", "Sequence of emojis", id="multi_emojis"),
    pytest.param("a ä ü", "Latin with diacritics", id="multi_diacritics"),
    pytest.param("test\n\t ", "String with various whitespace", id="multi_whitespace"),
]


@pytest.mark.parametrize("test_text, description", MULTILINGUAL_CASES)
def test_fast_tokenizer_multilingual_cases(multilingual_tokenizer, test_text, description):
    """
    Tests that the C++ tokenizer produces IDENTICAL results to the Python reference
    across a variety of real-world multilingual and multi-script strings.
    """
    python_tok, pretokenizer = multilingual_tokenizer
    cpp_tok = FastScriptTokenizer(
        merge_rules=python_tok.merge_rules, pretokenizer=pretokenizer, metadata=python_tok.metadata
    )

    python_tokens = python_tok.encode(test_text)
    cpp_tokens = cpp_tok.encode(test_text)

    assert cpp_tokens.tolist() == python_tokens.tolist(), (
        f"Multilingual mismatch for case: '{description}'\n"
        f"String: {repr(test_text)}\n"
        f"Python (correct): {python_tokens.tolist()}\n"
        f"C++    (buggy):   {cpp_tokens.tolist()}"
    )


def test_fast_tokenizer_error_handling():
    """Test error handling for incorrect pre-tokenizer types."""
    # FastScriptTokenizer only supports ScriptEncodingPretokenizer
    pretokenizer = get_pretokenizer("bytes_gpt4")
    with pytest.raises(RuntimeError, match="FastScriptTokenizer only supports ScriptEncodingPretokenizer"):
        FastScriptTokenizer(merge_rules=[], pretokenizer=pretokenizer, metadata={})
    
    # It must also have enforce_char_boundaries=True
    pretokenizer_no_cb = get_pretokenizer("scriptenc") # This is ScriptEncodingPretokenizer but with the flag set to False
    with pytest.raises(RuntimeError, match="enforce_char_boundaries"):
         FastScriptTokenizer(merge_rules=[], pretokenizer=pretokenizer_no_cb, metadata={})