import hashlib,inspect,json,platform,shutil
from pathlib import Path
import tokenizers
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,decoders
from scripts.course_experiments.text import _BPE,_utf8_prefix
from tiny_perceptron.data import SPECIALS
root=Path(__file__).resolve().parents[4];base=Path(__file__).resolve().parent
raw=root/'docs/course-experiments/results/tokenizer.json';m=json.loads(raw.read_text())
tok=Tokenizer(models.BPE());tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
trainer=trainers.BpeTrainer(vocab_size=280,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
tok.train_from_iterator(['貓看狗。','a cat sees a dog.'],trainer)
rows=[]
for text in ['小鳥🙂new','未見字🦊']:
 encoded=tok.encode(text);restored=tok.decode(encoded.ids);assert restored==text
 rows.append({'text':text,'tokens':encoded.tokens,'ids':encoded.ids,'restored':restored})
original=root/'checkpoints/course/tokenizer/tokenizer-bpe512.json'
expected=next(a['sha256'] for a in m['artifacts'] if a['path']=='tokenizer-bpe512.json')
assert hashlib.sha256(original.read_bytes()).hexdigest()==expected
bpe=_BPE(Tokenizer.from_file(str(original)));assert bpe.vocab_size==512
replayed=[]
for s in m['results']['roundtrip']:
 ids=bpe.encode(s['text']);restored=bpe.decode(ids)
 assert ids==s['ids'] and restored==s['restored']==s['text']
 replayed.append({'text':s['text'],'ids':ids,'restored':restored,'same':restored==s['text'],'control_ids':set(ids)&bpe.special_ids})
for row in replayed:row['control_ids']=list(row['control_ids'])
assert len(pre_tokenizers.ByteLevel.alphabet())==256
assert sum(m['results']['data'][s]['records'] for s in ['train','validation','test'])==192
shutil.copyfile(raw,base/'sources/tokenizer-result.json');shutil.copyfile(original,base/'sources/tokenizer-bpe512.json')
(base/'sources/tokenizers-bytelevel-docstrings.txt').write_text('\n\n'.join(inspect.getdoc(obj) for obj in [pre_tokenizers.ByteLevel,pre_tokenizers.ByteLevel.alphabet,trainers.BpeTrainer,decoders.ByteLevel]))
print(json.dumps({'python':platform.python_version(),'tokenizers':tokenizers.__version__,'device':'cpu','toy_vocab':tok.get_vocab_size(),'alphabet_size':256,'toy':rows,'historical_vocab':bpe.vocab_size,'historical_specials':{s:bpe.tokenizer.token_to_id(s) for s in SPECIALS},'original_tokenizer_sha256':expected,'raw_result_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'historical_splits':m['results']['data'],'historical_roundtrip_readback':replayed,'prefix_256_bytes_contract':{'input':'小'*86,'output_bytes':len(_utf8_prefix('小'*86,256).encode()),'output_codepoints':len(_utf8_prefix('小'*86,256))},'historical_environment':{k:m[k] for k in ['revision','device','torch_version','python_version','seed','gpu']},'current_script_hash_matches_original':hashlib.sha256((root/'scripts/course_experiments/text.py').read_bytes()).hexdigest()==m['code_sha256']['scripts/course_experiments/text.py']},ensure_ascii=False,indent=2))

