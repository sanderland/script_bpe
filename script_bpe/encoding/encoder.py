import functools
import itertools
import json
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Any

SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unicode_scripts.txt")
END_CODEPOINT = 0xE0FFF  # full cover of non private use code points, excluding surrogates


SCRIPTS_WHICH_USE_SPACES = [
    "Latin",  # Near space 18.0% of time, overall count 296,227,775,159
    "Arabic",  # Near space 18.6% of time, overall count 116,931,857,999
    "Devanagari",  # Near space 23.0% of time, overall count 3,391,578,438
    "Hangul",  # Near space 28.8% of time, overall count 2,456,037,316
    "Ethiopic",  # Near space 22.3% of time, overall count 493,107,008
    "Cyrillic",  # Near space 13.6% of time, overall count 119,738,736
    "Greek",  # Near space 17.2% of time, overall count 13,040,710
    "Hebrew",  # Near space 17.0% of time, overall count 6,081,243
    "Bengali",  # Near space 16.4% of time, overall count 3,630,267
    "Syriac",  # Near space 12.6% of time, overall count 2,669,016
    "Oriya",  # Near space 16.6% of time, overall count 1,147,764
    "Tamil",  # Near space 11.9% of time, overall count 1,084,075
    "Telugu",  # Near space 13.6% of time, overall count 694,309
    "Gurmukhi",  # Near space 22.8% of time, overall count 394,438
    "Gujarati",  # Near space 18.3% of time, overall count 388,983
    "Sinhala",  # Near space 17.5% of time, overall count 369,417
    "Malayalam",  # Near space 10.7% of time, overall count 339,796
    "Armenian",  # Near space 14.3% of time, overall count 338,586
    "Kannada",  # Near space 13.3% of time, overall count 326,104
    "Georgian",  # Near space 14.1% of time, overall count 277,463}
]


def char_name(c: str):
    return unicodedata.name(c, "<UNKNOWN>")


@functools.cache
def supercategory(category):
    sc = category[0]
    if sc in {"P", "S"}:
        return "PS"  # Punctuation/Symbol
    if sc in {"L", "M"}:
        return "LM"  # Letter/Non-spacing Mark (like accept modifiers)
    return sc


def unicode_script_map(filename=SCRIPTS_PATH) -> dict[str, dict[str, str]]:
    """
    Load Unicode script and category data from a file.

    Returns:
        A dictionary mapping codepoint (int) to a dict with 'script' and 'category' keys
    """
    char_info: dict[str, dict[str, str]] = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse 0000..001F    ; Common # Cc  [32] <control-0000>..<control-001F>
            range_str, semicol, script, hash, category, *_ = line.split()
            assert semicol == ";" and hash == "#", f"Unexpected format in line: {line}"
            # Handle single codepoint or range
            if ".." in range_str:
                start_str, end_str = range_str.split("..")
                start, end = int(start_str, 16), int(end_str, 16)
            else:
                start = end = int(range_str, 16)
            # Add each codepoint in the range to the result dictionary
            for cp in range(start, end + 1):
                char_info[chr(cp)] = dict(
                    script=script,
                    category=category,  # file is typically newer than python's unicodedata
                )
    for entry in char_info.values():
        entry["supercategory"] = supercategory(entry["category"])
    return char_info


class ScriptEncoding:
    VERSION = "1.0"
    FIRST_TOKEN_ID = 1  # leave 0 for padding
    SPLIT_SCRIPTS = [  # Large scripts split into sub-blocks
        ("Han", "LM"),  # 98687 entries
        ("Hangul", "LM"),  # 11677 entries
        ("Common", "PS"),  # 7195 entries
        ("Tangut", "LM"),  # 6914 entries
        ("Egyptian_Hieroglyphs", "LM"),  # 5089 entries
    ]
    SCRIPT_CAT_OVERRIDE = {
        "\n": ("Common", "Z"),  # Newline – whitespace
        "\t": ("Common", "Z"),  # Tab – whitespace
        "\u30fc": ("Inherited", "LM"),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
        "\uff70": ("Inherited", "LM"),  # ﾊﾟｰﾃｨｰ (halfwidth)
        "\u0640": ("Arabic", "LM"),  # ـــمــر (used in Arabic script shaping)
    }
    # pretokenize allows a single leading space for these
    DEFAULT_SCRIPT_CAT_WITH_SPACE = [(s, "LM") for s in SCRIPTS_WHICH_USE_SPACES] + [("Common", "PS")]

    def script_encoding_blocks(self):
        sc_map = unicode_script_map()

        chars_by_sc = defaultdict(list)
        num_chars_by_script = Counter()
        for c, char_info in sc_map.items():
            if c in self.SCRIPT_CAT_OVERRIDE:
                char_info["script"], char_info["supercategory"] = self.SCRIPT_CAT_OVERRIDE[c]
            chars_by_sc[(char_info["script"], char_info["supercategory"])].append(c)
            num_chars_by_script[char_info["script"]] += 1

        largest_block = max(chars_by_sc.items(), key=lambda kv: 0 if kv[0][:2] in self.SPLIT_SCRIPTS else len(kv[1]))
        num_index_tokens = len(largest_block[1])
        blocks = []

        for sc, cps in sorted(chars_by_sc.items(), key=lambda kv: (num_chars_by_script[kv[0][0]], kv[1]), reverse=True):
            sid = len(blocks) + num_index_tokens
            for sub_block, start in enumerate(range(0, len(cps), num_index_tokens)):
                blocks.append([sid, *sc, sub_block, "".join(cps[start : start + num_index_tokens])])

        config = dict(
            version=self.VERSION,
            num_index_tokens=num_index_tokens,
            num_blocks=len(blocks),
            script_cat_with_space=self.DEFAULT_SCRIPT_CAT_WITH_SPACE,
            stats=dict(  # just diagnostic
                largest_block=list(largest_block[0]),
            ),
            settings=dict(  # used to create, but not used downstream
                split_scripts=[list(sc) for sc in self.SPLIT_SCRIPTS],
            ),
        )
        return config, blocks

    def __init__(self):
        self.config, self.blocks = self.script_encoding_blocks()

    def export_config(self) -> dict[str, Any]:
        return dict(
            **self.config,
            blocks=self.blocks,
        )
