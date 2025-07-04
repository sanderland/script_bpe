"""Tests for FastScriptTokenizer C++ implementation."""

import pytest
import tempfile
import shutil
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.bpe.train import train_bpe
from script_bpe.bpe.fast.tokenizer import FastScriptTokenizer
from script_bpe.corpus import PretokenizedCorpus
from typing import cast


@pytest.fixture
def temp_dir():
    """Create a temporary directory for corpus storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def trained_tokenizer(temp_dir):
    """Create a trained tokenizer for testing."""
    pretokenizer = get_pretokenizer("scriptenc_cb")
    
    # Create corpus with diverse training data
    corpus = PretokenizedCorpus.from_texts(
        name="test_corpus",
        texts=[
            "Hello world!",
            "Script encoding works: 世界你好",
            "Testing with äöü ñ ç",
            "Numbers: 123 456 !@#",
            "Mixed: Здравствуй мир! שלום עולם!",
            "Simple test",
            "🌍🌎🌏",  # Emojis
            "αβγδε",  # Greek
            "АБВГДЕё",  # Cyrillic
        ],
        pretokenizer=pretokenizer,
        base_path=temp_dir
    )
    
    tokenizer = train_bpe(
        pretokenizer=pretokenizer,
        corpus=corpus,
        additional_vocab_size=20,
        num_workers=1,
        verbose=False
    )
    
    return tokenizer, pretokenizer


@pytest.fixture
def fast_tokenizer(trained_tokenizer):
    """Create a FastScriptTokenizer for testing."""
    tokenizer, pretokenizer = trained_tokenizer
    
    return FastScriptTokenizer(
        merge_rules=tokenizer.merge_rules,
        pretokenizer=pretokenizer,
        metadata=tokenizer.metadata
    ), tokenizer, pretokenizer


# Extended edge cases for comprehensive testing
EDGE_CASES = [
    "",  # Empty string
    " ",  # Just space
    "  ",  # Multiple spaces
    "\t",  # Tab
    "\n",  # Newline
    "\r\n",  # Windows line ending
    "a",  # Single character
    "A",  # Single uppercase
    "1",  # Single digit
    "!",  # Single punctuation
    "🌍",  # Single emoji
    "α",  # Single Greek
    "А",  # Single Cyrillic
    "ש",  # Single Hebrew
    "世界",  # Single CJK
    "a" * 1000,  # Very long string
    "🌍" * 100,  # Many emojis
    "αβγδε" * 50,  # Many Greek
    "АБВГДЕё" * 30,  # Many Cyrillic
    "世界你好" * 20,  # Many CJK
    "a b c d e f g h i j",  # Many single chars
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",  # All punctuation
    "0123456789",  # All digits
    "abcdefghijklmnopqrstuvwxyz",  # All lowercase
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # All uppercase
    "aA1!🌍αАש世界",  # Mixed single chars
    "Hello\nWorld\tTest",  # Mixed whitespace
    "Hello\r\nWorld\r\nTest",  # Mixed line endings
    "Hello\u0000World",  # Null bytes
    "Hello\u0001World",  # Control characters
    "Hello\u00A0World",  # Non-breaking space
    "Hello\u200BWorld",  # Zero-width space
    "Hello\uFEFFWorld",  # BOM
]


@pytest.mark.parametrize("test_text", EDGE_CASES)
def test_fast_tokenizer_exact_compatibility(fast_tokenizer, test_text):
    """Test that FastScriptTokenizer produces IDENTICAL results to BPETokenizer for all edge cases."""
    fast_tok, original_tok, pretokenizer = fast_tokenizer
    
    # Encode with both tokenizers
    original_tokens = original_tok.encode(test_text)
    fast_tokens = fast_tok.encode(test_text)
    
    # Token sequences must be IDENTICAL
    assert original_tokens.tolist() == fast_tokens.tolist(), \
        f"Token mismatch for text {repr(test_text)}:\n" \
        f"  Python: {original_tokens.tolist()}\n" \
        f"  C++:    {fast_tokens.tolist()}"
    
    # Decoding should also be identical
    original_decoded = original_tok.decode(original_tokens)
    fast_decoded = fast_tok.decode(fast_tokens)
    
    assert original_decoded == fast_decoded, \
        f"Decode mismatch for text {repr(test_text)}:\n" \
        f"  Python: {repr(original_decoded)}\n" \
        f"  C++:    {repr(fast_decoded)}"


def test_fast_tokenizer_basic_functionality(fast_tokenizer):
    """Test basic functionality of FastScriptTokenizer."""
    fast_tok, _, pretokenizer = fast_tokenizer
    
    # Test basic encoding and decoding
    test_text = "Hello world!"
    encoded = fast_tok.encode(test_text)
    decoded = fast_tok.decode(encoded)
    
    # Should be able to encode and decode
    assert isinstance(encoded, list) or hasattr(encoded, 'tolist')
    assert isinstance(decoded, str)
    assert len(encoded) > 0
    
    # Round-trip should work
    expected = pretokenizer.normalize(test_text)
    assert decoded == expected


def test_fast_tokenizer_error_handling():
    """Test error handling when C++ backend is not available or fails."""
    
    # Test that FastScriptTokenizer only supports ScriptEncodingPretokenizer
    pretokenizer = get_pretokenizer("bytes_gpt4")
    
    with pytest.raises(RuntimeError, match="FastScriptTokenizer only supports ScriptEncodingPretokenizer"):
        FastScriptTokenizer(
            merge_rules=[],
            pretokenizer=pretokenizer,
            metadata={}
        )
    
    # Test that it works with ScriptEncodingPretokenizer
    script_pretokenizer = get_pretokenizer("scriptenc_cb")
    fast_tokenizer = FastScriptTokenizer(
        merge_rules=[],
        pretokenizer=script_pretokenizer,
        metadata={}
    )


def test_fast_tokenizer_inheritance():
    """Test that FastScriptTokenizer properly inherits from BPETokenizer."""
    
    pretokenizer = get_pretokenizer("scriptenc_cb")
    
    fast_tokenizer = FastScriptTokenizer(
        merge_rules=[],
        pretokenizer=pretokenizer,
        metadata={}
    )
    
    # Test that it has the expected methods and attributes
    assert hasattr(fast_tokenizer, 'encode')
    assert hasattr(fast_tokenizer, 'decode')
    assert hasattr(fast_tokenizer, 'pretokenizer')
    assert hasattr(fast_tokenizer, 'merge_rules')
    assert hasattr(fast_tokenizer, 'metadata')
    
    # Test that decode falls back to Python implementation
    from script_bpe.utils import token_array
    test_tokens = token_array([1, 2, 3])
    decoded = fast_tokenizer.decode(test_tokens)
    assert isinstance(decoded, str) 