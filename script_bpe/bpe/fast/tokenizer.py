 
import unicodedata
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
        # Convert script encoding to C++ format: char -> CharSCRIPTEnc
        cpp_script_encoding = {}
        for c, (sid, token_pair) in self.pretokenizer.script_encoding.items():
            token_id = self._merge_rules_dict.get(token_pair, (0, -1))[1]
            cpp_script_encoding[c] = CharSCRIPTEnc(sid, *token_pair, token_id)

        self._cpp_fast_tokenizer = FastTokenizer(
            cpp_script_encoding,
            self._merge_rules_dict,
        )
    
    def encode(self, text: str) -> TokenSeq:
        normalized = unicodedata.normalize('NFC', text)
        encoded_tokens = self._cpp_fast_tokenizer.encode(normalized)
        return token_array(encoded_tokens)
