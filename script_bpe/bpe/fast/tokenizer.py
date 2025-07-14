 
import unicodedata

import numpy as np
from script_bpe.bpe.tokenizer import BPETokenizer
from script_bpe.pretokenize import BasePretokenizer, ScriptEncodingPretokenizer
from script_bpe.utils import TokenSeq, token_array
from .fast_tokenizer_cpp import FastTokenizer, CharSCRIPTEnc

class FastScriptTokenizer(BPETokenizer):
    """Fast C++ implementation of BPE inference, for SCRIPT encoded with character boundaries."""
    
    def __init__(self, merge_rules, pretokenizer: BasePretokenizer, metadata=None):
        # Only support ScriptEncodingPretokenizer for full C++ implementation
        if not isinstance(pretokenizer, ScriptEncodingPretokenizer) or not pretokenizer.enforce_char_boundaries:
            raise RuntimeError("FastScriptTokenizer only supports ScriptEncodingPretokenizer with enforce_char_boundaries")
        super().__init__(merge_rules, pretokenizer, metadata)

        self._setup_cpp_backend()


    def _setup_cpp_backend(self):
        assert isinstance(self.pretokenizer, ScriptEncodingPretokenizer) # make type checker happy
        # Find max codepoint to size the vector
        max_cp = max(ord(c) for c in self.pretokenizer.script_encoding)
        cpp_script_encoding = [CharSCRIPTEnc(-1, -1, -1, -1) for _ in range(max_cp + 1)]
        for c, (sid, token_pair) in self.pretokenizer.script_encoding.items():
            token_id = self._merge_rules_dict.get(token_pair, (0, -1))[1]
            cpp_script_encoding[ord(c)] = CharSCRIPTEnc(token_id, sid, *token_pair)

        self._cpp_fast_tokenizer = FastTokenizer(
            cpp_script_encoding,
            {k: v[1] for k, v in self._merge_rules_dict.items()},
        )
    
    def encode(self, text: str) -> np.ndarray:
        normalized = unicodedata.normalize('NFC', text)
        return self._cpp_fast_tokenizer.encode(normalized)
