import hashlib,json,platform
from pathlib import Path
import tokenizers
from tokenizers import Tokenizer
from tiny_perceptron.data import ByteTokenizer,SPECIALS
from scripts.course_experiments.text import _BPE
root=Path(__file__).resolve().parents[4];base=Path(__file__).resolve().parent
tok=ByteTokenizer();rows=[]
for text in ['<assistant>','<image>']:
 ids=tok.encode(text);assert min(ids)>=8 and not set(ids)&set(range(8));assert tok.decode(ids)==text
 rows.append({'text':text,'ids':ids,'restored':tok.decode(ids),'control_ids':list(set(ids)&set(range(8)))})
assert rows[0]['ids'][0]==68 and rows[0]['ids'][-1]==70
assert len(rows[1]['ids'])==7 and tok.image_id==5
m=json.loads((base/'sources/tokenizer-result.json').read_text());original=base/'sources/tokenizer-bpe512.json'
raw=Tokenizer.from_file(str(original));bpe=_BPE(raw);probe=m['results']['roundtrip'][2]
ids=bpe.encode(probe['text']);assert ids==probe['ids'] and not set(ids)&bpe.special_ids and bpe.decode(ids)==probe['text']
default=raw.encode(probe['text']).ids
other=[]
for text in SPECIALS:
 values=bpe.encode(text);assert not set(values)&bpe.special_ids and bpe.decode(values)==text
 other.append({'literal':text,'ids':values,'restored':bpe.decode(values),'contains_control_id':bool(set(values)&bpe.special_ids)})
print(json.dumps({'python':platform.python_version(),'tokenizers':tokenizers.__version__,'device':'cpu','specials':list(enumerate(SPECIALS)),'byte_content_range':[min(i+8 for i in range(256)),max(i+8 for i in range(256))],'byte_literals':rows,'historical_probe':{'text':probe['text'],'ids':ids,'restored':bpe.decode(ids),'contains_control_id':bool(set(ids)&bpe.special_ids)},'default_bpe_recognizes_special_id':{'ids':default,'contains_user_id':bpe.user_id in default},'saved_bpe_all_special_literals':other,'saved_bpe_sha256':hashlib.sha256(original.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))

