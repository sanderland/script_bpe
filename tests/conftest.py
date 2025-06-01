import pytest

from script_bpe.pretokenize import get_pretokenizer


@pytest.fixture
def taylorswift_text():
    with open("tests/data/taylorswift.txt", "r") as f:
        return f.read()


@pytest.fixture
def script_encoding_pretokenizer():
    return get_pretokenizer("scriptenc")


@pytest.fixture
def script_encoding_nosplit_pretokenizer():
    return get_pretokenizer("scriptenc_nosplit_cb")
