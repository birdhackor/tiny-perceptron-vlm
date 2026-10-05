"""Bounded independent review: original fence variations and recorded split/denominator reconstruction.
No model is constructed; no fitting, inference, network, or dataset downloads.
"""
import ast
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
import torch

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-5_12-independent'
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import ByteTokenizer, shifted, fingerprint

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
 'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu',
 'model_constructed':'false','training_executed':'false','source_mode':'existing original inputs',
 'environment_names':{k:os.environ.get(k,'unset') for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
(OUT/'cpu-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')

def digest(raw):return hashlib.sha256(raw).hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

fence=(OUT/'extraction/fence-1.py').read_bytes()
ns={};exec(compile(fence,'original fence-1.py','exec'),ns)
grams=ns['grams']
sentences=['紅色圓形在左邊','紅色圓形在右邊']
assert ns['score']==0.5 and len(ns['common'])==4 and len(ns['all_parts'])==8
modified=fence.decode().replace('grams("紅色圓形在左邊")','grams("紅色圓形在左邊", n=3)').replace('grams("紅色圓形在右邊")','grams("紅色圓形在右邊", n=3)').encode()
(OUT/'code/fence-n3.py').write_bytes(modified)
ns3={};exec(compile(modified,'fence-n3.py','exec'),ns3)
assert ns3['common']=={'紅色圓','色圓形','圓形在'}
assert len(ns3['a'])==len(ns3['b'])==5 and len(ns3['all_parts'])==7
assert ns3['score']==3/7 and abs(ns3['score']-0.429)<0.0005
edge=[]
for left,right in [('', ''),('左','右')]:
 try:score=len(grams(left)&grams(right))/len(grams(left)|grams(right))
 except ZeroDivisionError:edge.append({'inputs':[left,right],'observed':'ZeroDivisionError'})
 else:raise AssertionError('expected undefined empty-set denominator')
synonyms=['汽車停止','轎車不再前進']
assert len(grams(synonyms[0])&grams(synonyms[1]))==0
assert grams('人人')==grams('人人人')=={'人人'}
toy={'n2':{'grams':list(map(lambda x:sorted(grams(x)),sentences)),'score':ns['score']},
 'n3':{'grams':list(map(lambda x:sorted(grams(x,3)),sentences)),'score':ns3['score']},
 'changed_n_is_different_score':True,'left_right_have_distinct_full_hash':fingerprint(sentences[0])!=fingerprint(sentences[1]),
 'repeated_grams_unique':sorted(grams('人人人')),'empty_edges':edge,
 'semantic_rewording_zero_overlap':{'inputs':synonyms,'jaccard':0},
 'different_short_sentences_equal_gram_sets':{'inputs':['人人','人人人'],'jaccard':1}}

report=json.loads((OUT/'inputs/docs/course-experiments/results/real_text.json').read_text())
historical={};namespace={'json':json,'hashlib':hashlib,'random':random,'Path':Path,
 'write_json':write_json,'ByteTokenizer':ByteTokenizer,'shifted':shifted,'torch':torch}
for source,names in [('scripts/course_experiments/text.py',{'_json_bytes','_digest','_deduplicate_text','_save_splits'}),
 ('scripts/course_experiments/common.py',{'split_records','records_sha256','text_examples'})]:
 path=OUT/'inputs/historical'/source;raw=path.read_bytes()
 assert digest(raw)==report['code_sha256'][source]
 historical[source]={'sha256':digest(raw),'matches_report_code_sha256':True,'executed_functions':sorted(names)}
 tree=ast.parse(raw)
 selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
 assert {node.name for node in selected}==names
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
dedup,split,save,text_examples=map(namespace.get,('_deduplicate_text','split_records','_save_splits','text_examples'))
class Context:output=OUT/'recomputed-splits'
ctx=Context();ctx.output.mkdir(exist_ok=True)
runs={}
for identifier,filename in [('tinystories','tinystories-train-512.jsonl'),('chinese-poetry','chinese-classical-train-365.jsonl')]:
 input_path=OUT/'inputs/data/training/text-initial'/filename
 raw=[json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
 original=report['results']['runs'][identifier]
 registered=next(a for a in report['training_assets'] if a['id']==identifier) if 'training_assets' in report else next(a for a in report['assets'] if a['id']==identifier)
 entry=next(f for f in registered['files'] if Path(f['path']).name==filename)
 assert digest(input_path.read_bytes())==entry['sha256'] and input_path.stat().st_size==entry['bytes']
 records=dedup(raw);parts=split(records,seed=report['seed']);manifest=save(ctx,parts,name=identifier+'-data')
 assert len(raw)==original['source_records'] and len(records)==original['deduplicated_records']
 assert manifest==original['data']
 families={name:{row['family'] for row in rows} for name,rows in parts.items()}
 overlap={left+'-'+right:len(families[left]&families[right]) for left,right in [('train','validation'),('train','test'),('validation','test')]}
 assert set(overlap.values())=={0}
 assert all(r['family']==fingerprint(r['text']) for r in records)
 assert all(after['text']==before['text'] for before,after in zip(raw,records,strict=True))
 assert namespace['records_sha256'](parts['train'])==original['training']['records_sha256']
 denominators={}
 for name in ['validation','test']:
  rows=parts[name];examples=text_examples(rows,max_length=128)
  bytecount=sum(len(r['text'].encode()) for r in rows)
  targetcount=sum(y.numel() for x,y in examples)
  assert targetcount==bytecount+len(rows) # one EOS target per record, BOS only input
  for timing in ['before','after']:
   recorded=original[timing][name]
   assert recorded['records']==len(rows) and recorded['examples']==len(examples)
   assert recorded['effective_tokens']==targetcount and recorded['raw_utf8_bytes']==bytecount
   assert abs(recorded['nll']-recorded['nll_sum']/targetcount)<1e-12
   assert abs(recorded['bpb_including_eos_boundary_targets']-recorded['nll_sum']/(bytecount*math.log(2)))<1e-12
  denominators[name]={'records':len(rows),'windows':len(examples),'target_tokens':targetcount,
   'utf8_bytes':bytecount,'before_after_denominators_identical':True,'nll_ratio_checked':'1e-12 absolute tolerance'}
 runs[identifier]={'input_sha256':digest(input_path.read_bytes()),'source_records':len(raw),
 'deduplicated_records':len(records),'recomputed_manifest':manifest,'full_fingerprint_cross_split_overlap':overlap,
 'family_equals_complete_normalized_text_sha256':True,'original_text_preserved':True,
 'denominators':denominators,'split_warning':original['split_warning'],'steps_recorded_only':original['training']['steps']}

source_poems=json.loads((OUT/'inputs/data/training/text-initial/tang300-source.json').read_text())
assert len(source_poems)==366
poem_renderings=[row['title']+'\n'+row['author']+'\n'+'\n'.join(row['paragraphs']) for row in source_poems]
assert len({digest(x.encode()) for x in poem_renderings})==365
log=json.loads((OUT/'inputs/data/training/text-initial/chinese-poetry-dedup-log.json').read_text())
assert len(log)==1 and log[0]['source_record_index']==124
assert digest(poem_renderings[124].encode())==log[0]['text_sha256']
prefix=(OUT/'inputs/data/training/text-initial/tinystories-train-prefix-complete.txt').read_text()
prefix_stories=[s.strip() for s in prefix.split('<|endoftext|>') if s.strip()]
stored=[json.loads(line)['text'] for line in (OUT/'inputs/data/training/text-initial/tinystories-train-512.jsonl').read_text().splitlines()]
assert prefix_stories==stored and len(stored)==512

toy_rows=[{'text':sentences[0],'answer':'left','family':'position-family'},
 {'text':sentences[1],'answer':'right','family':'position-family'}]+[
 {'text':str(i),'answer':str(i),'family':'other-'+str(i)} for i in range(4)]
toy_parts=split(toy_rows)
place=[name for name,rows in toy_parts.items() if any(r['family']=='position-family' for r in rows)]
assert len(place)==1
assert {r['answer'] for r in toy_parts[place[0]] if r['family']=='position-family'}=={'left','right'}
write_json(OUT/'inspection-results.json',{'toy':toy,'historical_code':historical,'runs':runs,
 'upstream_poetry_count':366,'prepared_poetry_count':365,'original_exact_duplicate_removed':1,
 'complete_tinystories_prefix_count':512,'toy_family_preserves_distinct_labels':{'split':place[0],'labels':['left','right']},
 'limits':'Reconstructed original full-normalized-content grouping and JSON denominator/hash contracts only. No near-duplicate clustering or fresh model performance measured.'})
print(json.dumps({'toy':toy,'runs':runs,'all_assertions_passed':True},ensure_ascii=False,indent=2))
