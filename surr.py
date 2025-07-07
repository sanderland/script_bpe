#%%
from script_bpe.encoding.encoder import ScriptEncoding
from script_bpe.utils import is_unassigned_private_surrogate
import unicodedata

def main():
	se = ScriptEncoding()
	found = []
	# Collect all characters in ScriptEncoding
	all_chars = set()
	for block in se.blocks:
		# block format: [sid, script, supercategory, sub_block, chars]
		chars = block[4]
		script = block[1]
		supercat = block[2]
		for c in chars:
			all_chars.add((c, script, supercat))
	for c, script, supercat in sorted(all_chars):
		if is_unassigned_private_surrogate(c):
			found.append((c, script, supercat))
	for c, script, supercat in found:
		ucat = unicodedata.category(c)
		print(f"U+{ord(c):04X} {repr(c)} script={script} supercat={supercat} unicodedata category={ucat}")
	print(f"Total: {len(found)}")

if __name__ == "__main__":
	main()
