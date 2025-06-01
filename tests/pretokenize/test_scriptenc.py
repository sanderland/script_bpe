import json

import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import TokenSeq, token_array


def test_se_tokenize(script_encoding_pretokenizer):
    text = "abc def"
    tokenized = script_encoding_pretokenizer.encode_and_chunk(text)
    assert isinstance(tokenized, list)
    assert all(isinstance(group, TokenSeq) for group in tokenized)

    token_ids = sum(tokenized, token_array([]))
    detokenized = script_encoding_pretokenizer.decode(token_ids)
    assert isinstance(detokenized, str)
    assert detokenized == text


TEST_CASES = [
    # 1. Latin "Hello ", Arabic "العالمية", spaces, Emoji "❤️"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064A  \u2764", 4),
    # 2. Latin "Hello ", Arabic "العالمية", ZWJ+Emoji "‍❤️"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064A\u200C❤️", 3),
    # 3. Latin "Hello ", Arabic "العالمية", space, Emoji "🌍"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064A \u0020🌍", 4),
    # 4. Latin "abc ", Arabic "مَبْنِي" (with diacritics), spaces "  ", Latin "xyz"
    ("abc \u0645\u0650\u0628\u0646\u064A  xyz", 4),
    # 5. Arabic "رينر" (with diacritic), space+Emoji "❤️"
    ("\u0631\u064A\u0646\u0650\u0631 ❤️", 2),
    # 6. Pure Arabic text only
    ("\u0627\u0639\u062F\u0644\u0639\u0645\u062F\u0631\u0648\u0632", 1),
    # 7. Tamil "வணக்கம்", space, Emoji "❤️"
    ("வணக்கம் ❤️", 2),
    # 8. Mixed chinese
    ("欢迎歡迎来到來到中国中國一起学习學習古字𠀋𠀍𠀎", 1),
    # 9. Numbers do not combine with spaces
    ("1234 5678", 3),
    # 10. Punctuation does combine
    ("x = a + b", 5),
    # 11. Japanese: Han -> Hira -> Kata (Should all merge into one chunk)
    ("漢字ひらがな", 1),
    # 12. Japanese: Hira -> Han -> Kata (Should all merge into one chunk)
    ("ひらがな漢字", 1),
    # 18. Japanese with Space
    (" 日本語 ", 3),  # Space, JPN(Han+Hira), Space
    # 19. Japanese with Number (Should split)
    ("第1番", 3),  # JPN(Han), Number, JPN(Han)
]


@pytest.mark.parametrize("text, n_expected", TEST_CASES)
def test_se_pretokenize_groups(script_encoding_pretokenizer, text, n_expected):
    script_encoding_pretokenizer = get_pretokenizer("scriptenc_cb")
    pretokenized_groups = script_encoding_pretokenizer.encode_and_chunk(text)
    assert (
        len(pretokenized_groups) == n_expected
    ), f"Expected {n_expected} groups but found {len(pretokenized_groups)} for {text!r}, got {len(pretokenized_groups)}"

    # detokenize and check if we get the original string back
    detokenized = script_encoding_pretokenizer.decode(sum(pretokenized_groups, token_array([])))
    assert detokenized == text, f"Detokenized string {detokenized!r} does not match original {text!r}"


@pytest.mark.parametrize("text", [text for text, _ in TEST_CASES])
def test_nosplit_pretokenizer(text, script_encoding_nosplit_pretokenizer):
    pretokenized_groups = script_encoding_nosplit_pretokenizer.encode_and_chunk(text)
    assert len(pretokenized_groups) == 1
    assert len(pretokenized_groups[0]) == len(text) * 2


def test_se_hash_identical(script_encoding_pretokenizer):
    pretokenizer1 = get_pretokenizer("scriptenc")
    assert pretokenizer1.hash() == script_encoding_pretokenizer.hash()
    print(pretokenizer1.hash())


def test_se_can_json(script_encoding_pretokenizer):
    json.dumps(script_encoding_pretokenizer.config)
