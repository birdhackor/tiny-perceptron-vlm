import inspect,json,platform
from pathlib import Path
import tokenizers
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
tok=Tokenizer(models.BPE(unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace()
tok.train_from_iterator(['hello hello hello'],trainers.BpeTrainer(vocab_size=20,special_tokens=['[UNK]'],show_progress=False))
whole=tok.encode('hello').ids;chunks=tok.encode('hel').ids+tok.encode('lo').ids
assert len(whole)==1 and whole!=chunks and len(chunks)>=2
spaced=tok.encode('hello hello').ids;safe=tok.encode('hello ').ids+tok.encode('hello').ids;assert spaced==safe
base=Path(__file__).resolve().parent;(base/'sources/tokenizers-whitespace-docstring.txt').write_text(inspect.getdoc(pre_tokenizers.Whitespace))
print(json.dumps({'python':platform.python_version(),'tokenizers':tokenizers.__version__,'device':'cpu','actual_vocab_size':tok.get_vocab_size(),'whole_hello':whole,'midword_chunks_hel_lo':chunks,'whole_hello_tokens':tok.encode('hello').tokens,'hel_tokens':tok.encode('hel').tokens,'lo_tokens':tok.encode('lo').tokens,'whole_hello_hello':spaced,'space_boundary_chunks':safe,'space_boundary_same':spaced==safe,'pretokenize_hello_hello':tok.pre_tokenizer.pre_tokenize_str('hello hello'),'unknown_example':tok.encode('z').tokens,'models_trained':'tokenizer rules only'},ensure_ascii=False,indent=2))

