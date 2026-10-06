"""Same-owner 17.9 callback: unchanged proof hashes, links, and bounded CLI contract."""
import ast
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OLD = ROOT/'docs/technical-reviews/artifacts/phase4-17_9-independent'
TASK = '/root/phase4_factual_coordinator/factual_17_9'
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def emit(name,value):
    print(json.dumps({'check':name,'result':value},ensure_ascii=False))

torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None
environment={'python':platform.python_version(),'torch':str(torch.__version__),
 'torch_git':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),
 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}
emit('environment',environment)
prior_path=BASE/'prior-owner-report.opaque.json'
prior=json.loads(prior_path.read_bytes())
assert prior['reviewer_task']==TASK
checks=[]
for group in ('sources','artifacts'):
    for record in prior[group]:
        if 'path' in record and 'sha256' in record:
            actual=sha((ROOT/record['path']).read_bytes())
            assert actual==record['sha256'],record['id']
            checks.append({'group':group,'id':record['id'],'path':record['path'],'sha256':actual})
emit('unchanged_prior_evidence',{'checked_file_hashes':len(checks),'all_match':True,
 'scope':'Original PTQ concepts, exact width8 fence and 4/8-bit CPU proof, raw existing measurements and source versions; no changed claim content.'})

spec=importlib.util.spec_from_file_location('actual_section_helper',ROOT/'docs/review-tools/section_facts.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
current,_,first=helper.original_section(ROOT/'course/chapters/17.md','17.9')
frozen=(OLD/'section.md').read_bytes()
assert sha(current)=='1869df773b6250f25f1bbf2e377f784c2181834ba8bab856e191975c6677e5ae'
old_phrase='結果見[17.15](17.md#17.15)'.encode()
new_phrase='品質檢查與實報入口見[17.15](17.md#17.15)'.encode()
assert frozen.count(old_phrase)==1 and frozen.replace(old_phrase,new_phrase)==current
old_fences=helper.fences(frozen,first);new_fences=helper.fences(current,first)
assert len(old_fences)==len(new_fences)==1
assert all(a['raw']==b['raw'] and a['info']==b['info'] for a,b in zip(old_fences,new_fences,strict=True))
context,_,context_line=helper.original_section(ROOT/'course/chapters/17.md','17.15')
old_context,_,_=helper.original_section(OLD/'frozen-17.md','17.15')
assert context==old_context
assert '品質檢查要與模型檔案大小、記憶體和速度一起報告。'.encode() in context
assert '[量化實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/quantization.json)'.encode() in context
assert '[T.9](../training.md#T.9)'.encode() in context
t9,_,t9line=helper.original_section(ROOT/'course/training.md','T.9')
assert b'--split-label test' in t9 and b'--limit all' in t9
emit('section_and_link_diff',{'current_source_sha256':sha(current),'owner_frozen_source_sha256':sha(frozen),
 'only_change':{'before':old_phrase.decode(),'after':new_phrase.decode()},'new_or_changed_17_9_commands':[],
 'python_fence_sha256':sha(new_fences[0]['raw']),'python_fence_unchanged':True,
 'linked_17_15_sha256':sha(context),'linked_17_15_unchanged_against_owner_frozen_chapter':True,
 'linked_contract':'17.15 explains same-input quality comparisons and links to raw report and T.9 operations.',
 'T9_sha256':sha(t9),'section_first_line':first,'context_first_line':context_line,'T9_first_line':t9line})

code=(ROOT/'scripts/evaluate.py').read_bytes()
tree=ast.parse(code)
locations={n.name:{'line':n.lineno,'end':n.end_lineno} for n in tree.body if isinstance(n,ast.FunctionDef)}
emit('actual_cli_source',{'path':'scripts/evaluate.py','sha256':sha(code),'AST_locators':locations,
 'inspected_contract':'parse_limit(all)->None; main selects all records; split-label is declared metadata; evaluate validates SFT labels, records IDs/row/status and separates exact content/EOS completion.'})

records=[{'messages':[{'role':'user','content':'a'},{'role':'assistant','content':'x'}]},
         {'messages':[{'role':'user','content':'b'},{'role':'assistant','content':'yz'}]}]
dataset_raw=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)).encode()
(BASE/'callback-toy-data.jsonl').write_bytes(dataset_raw)
torch.manual_seed(0)
model=TinyLM(ModelConfig(width=8,layers=1,max_length=32,tied=False)).eval()
commands=[]
checkpoint_sha=None
tok=ByteTokenizer()
with tempfile.TemporaryDirectory(prefix='phase4-17_9-cli-') as tmp:
    temp=Path(tmp)
    checkpoint=temp/'untrained-width8.pt'
    torch.save({'format_version':1,'config':asdict(model.config),'model':model.state_dict(),
                'optimizer':None,'tokenizer':tok.state()},checkpoint)
    checkpoint_sha=sha(checkpoint.read_bytes())
    data=temp/'bounded-data.jsonl';data.write_bytes(dataset_raw)
    outputs=[]
    for label in ('validation','test'):
        output=temp/('report-'+label+'.json')
        argv=[str(ROOT/'.venv/bin/python'),'scripts/evaluate.py',str(checkpoint),
              '--data',str(data),'--mode','sft','--tokens','3','--limit','all','--device','cpu',
              '--output',str(output)]
        if label=='test':argv += ['--split-label','test']
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=30,check=False)
        commands.append({'argv':argv,'exit_code':result.returncode})
        (BASE/('cli-'+label+'.stdout.txt')).write_bytes(result.stdout)
        (BASE/('cli-'+label+'.stderr.txt')).write_bytes(result.stderr)
        assert result.returncode==0,result.stderr.decode()
        raw=output.read_bytes();report=json.loads(raw)
        (BASE/('cli-'+label+'.raw.json')).write_bytes(raw)
        assert report['declared_split']==label
        assert report['records_read']==report['records_selected']==report['loss_evaluated_records']==report['generation_evaluated_records']==2
        assert report['unselected_records']==report['generation_skipped_records']==report['loss_skipped_records']==0
        assert report['limit']=='all' and report['effective_tokens']==5
        assert report['metric_denominators']['exact_match']==report['metric_denominators']['completed_exact_match']==2
        assert [s['row'] for s in report['samples']]==[0,1]
        for sample in report['samples']:
            ids=sample['generated_ids']
            assert len(ids)<=3
            ended=bool(ids and ids[-1]==tok.eos_id and ids.count(tok.eos_id)==1)
            content=ids[:-1] if ended else ids
            hit=(content==tok.encode(sample['target']) and not sample['invalid_special_tokens'])
            assert sample['exact_match']==hit and sample['completed_exact_match']==(hit and ended)
        outputs.append(report)
        emit('bounded_cli_'+label,{'declared_split':label,'records_read':2,'records_selected':2,
          'loss_and_generation_evaluated':2,'effective_answer_targets_including_eos':5,'max_new_tokens':3,
          'rows':[s['row'] for s in report['samples']],
          'generated_ids':[s['generated_ids'] for s in report['samples']],
          'statuses':[s['generation_status'] for s in report['samples']],
          'raw_json_sha256':sha(raw),'raw_data_sha256':sha(dataset_raw),
          'purpose':'Route/argument/denominator/status contract only; random model outputs are not capability scores.'})
    assert outputs[0]['samples']==outputs[1]['samples']
    assert sha(checkpoint.read_bytes())==checkpoint_sha
assert not checkpoint.exists()
emit('cleanup_and_scope',{'retained_neural_weight_files':0,'unchanged_ephemeral_checkpoint_sha256':checkpoint_sha,
 'metadata_label_changes_do_not_change_samples':True,'full_training_executed':False,
 'existing_model_reevaluation':False,'long_recipe_executed':False,'GPU_used':False,
 'page_render_required':False,'visual_scope':'No figure or visual-layout change; textual link destination personally read, no root parity used.'})

receipt={'reviewer_task':TASK,'review_kind':'same-original-owner-command-callback',
 'source':'course/chapters/17.md#17.9','source_sha256':sha(current),
 'prior_opaque_report':{'path':prior_path.relative_to(ROOT).as_posix(),'sha256':sha(prior_path.read_bytes())},
 'environment':environment,'prior_issues_count':len(prior['issues']),'prior_history_preserved':'complete prior report bytes',
 'reused_files':checks,'new_commands_in_17_9':[],
 'bounded_linked_T9_contract_commands':commands,
 'original_fence_unchanged_sha256':sha(new_fences[0]['raw']),
 'linked_contexts':{'17.15':{'sha256':sha(context),'line':context_line,'same_as_owner_frozen_context':True},
                    'T.9':{'sha256':sha(t9),'line':t9line,'scope':'read-only operation contract, bounded substitute for evaluation flags'}},
 'ephemeral_input':{'shape_config':asdict(model.config),'initialization_seed':0,'checkpoint_sha256':checkpoint_sha,
                  'data_sha256':sha(dataset_raw),'data_records':2,'weight_retention':False},
 'verification':'All source/fence/diff/CLI/denominator assertions passed; no training or existing model quality remeasurement.'}
(BASE/'callback-execution-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
emit('callback_completed',{'assertions_passed':True,'receipt':'callback-execution-receipt.json',
                          'receipt_sha256':sha((BASE/'callback-execution-receipt.json').read_bytes())})
