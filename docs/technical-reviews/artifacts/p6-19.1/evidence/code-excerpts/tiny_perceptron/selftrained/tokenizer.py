tiny_perceptron/selftrained/tokenizer.py:L1-L12
1: """Train-side character vocabulary and program-inserted roles for the finite assistant."""
2: 
3: import hashlib
4: import json
5: from pathlib import Path
6: 
7: SPECIAL_NAMES = ("pad", "bos", "eos", "user", "assistant", "system", "tool", "image", "ocr", "audio", "unk")
8: SPECIALS = tuple(f"<{name}>" for name in SPECIAL_NAMES)
9: OCR_CHARACTERS = "大小上下左右開關入出人口"
10: 
11: 
12: class CharacterTokenizer:

tiny_perceptron/selftrained/tokenizer.py:L43-L56
43:     def decode(self, ids, skip_special_tokens=True):
44:         pieces = []
45:         for token in ids:
46:             token = int(token)
47:             if not 0 <= token < self.vocab_size:
48:                 raise ValueError("Token ID outside this tokenizer's vocabulary")
49:             if token == self.unk_id:
50:                 pieces.append("�")
51:             elif token in self.control_ids:
52:                 if not skip_special_tokens:
53:                     pieces.append(SPECIALS[token])
54:             else:
55:                 pieces.append(self.characters[token - len(SPECIALS)])
56:         return "".join(pieces)

tiny_perceptron/selftrained/tokenizer.py:L80-L95
80: def generation_report(tokenizer, ids):
81:     """Do not silently discard generated role markers and present them as valid answers."""
82:     ids = [int(token) for token in ids]
83:     invalid = [
84:         {"position": i, "id": token, "token": SPECIALS[token]}
85:         for i, token in enumerate(ids)
86:         if token in tokenizer.special_ids and token != tokenizer.eos_id
87:     ]
88:     answer_ids = ids[: ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids
89:     return {
90:         "answer": tokenizer.decode(answer_ids, skip_special_tokens=False),
91:         "generated_ids": ids,
92:         "eos": tokenizer.eos_id in ids,
93:         "invalid_special_tokens": invalid,
94:         "valid_answer_tokens": not invalid,
95:     }
