"""Small offline tokenizer pairing check; no language-model training or saved weights."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers
from tiny_perceptron.data import SPECIALS
from tiny_perceptron.tokenization import load_tokenizer

with tempfile.TemporaryDirectory(prefix="phase4-t_4-bpe-") as temp:
    path=Path(temp)/"paired.json"
    t=Tokenizer(models.BPE())
    t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder=decoders.ByteLevel()
    t.train_from_iterator(["circle circle red red shape shape", "圓🙂 <user> circle"],trainers.BpeTrainer(vocab_size=280,special_tokens=list(SPECIALS),initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    t.save(str(path))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    vocab=t.get_vocab_size()
    bound={"tokenizer":{"type":"bpe","vocab_size":vocab,"sha256":digest,"specials":list(SPECIALS)}}
    tok=load_tokenizer(path,vocab,bound)
    roundtrips=[]
    for text in ["圓🙂", "literal <user>", " circle "]:
        ids=tok.encode(text)
        assert tok.decode(ids)==text and all(i>=8 for i in ids)
        roundtrips.append({"text":text,"ids":ids,"restored":tok.decode(ids),"control_ids_absent":True})
    rejected={}
    for label,definition in [("missing-json",None),("wrong-json-hash",path)]:
        incompatible=bound if label=="missing-json" else {"tokenizer":{"type":"bpe","sha256":"0"*64}}
        try:
            load_tokenizer(definition,vocab,incompatible)
            raise AssertionError(label)
        except ValueError as error:
            rejected[label]=str(error)
    result={"environment":{"python":sys.version.split()[0],"tokenizers":version("tokenizers"),"device":"CPU tokenizer only"},"vocab_size":vocab,"temporary_json_sha256":digest,"paired_json_accepted":True,"roundtrips":roundtrips,"rejected":rejected,"permanent_tokenizer_json_or_weights_saved":False,"scope":"Tiny locally trained vocabulary probes the load contract; no language model or capability training."}
(OUT/"bpe-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False))
