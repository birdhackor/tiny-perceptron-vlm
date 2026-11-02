"""必要前提精確切片與第20章原始驗證紀錄的有界重算。"""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import unicodedata

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.model import TinyLM,ModelConfig

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())

def group(row):
    if row['task']=='scene':return 'photo_summary' if row['id'].endswith('/scene') else 'photo_fact'
    return {'chat':'text_chat','text_presence':'text_presence','ocr':'single_ocr','ocr_order':'ordered_ocr','typed_chat':'voice_typed_reference_chat','speech_chat':'voice_actual_asr_chat'}[row['task']]

def complete(row):
    ids=row['generated_token_ids']
    return bool(ids and len(ids)<=384 and ids[-1] in row['eos_token_ids'] and row['generated_tokens']==len(ids) and row['ended_with_eos'] is True and row['stop_reason']=='eos' and row['truncated'] is False and row['completion_unknown'] is False)

if __name__=='__main__':
    torch.set_num_threads(1)
    frozen=load(ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json')
    slices=[]
    for ident in ['20.8','20.12','20.13','first-steps','validation']:
        p=next(x for x in frozen['pages'] if x['page_id']==ident)
        raw=(ROOT/p['snapshot']).read_bytes()
        full=(ROOT/p['source']).read_text()
        if p['selector'].startswith('section:'):
            section=p['selector'].split(':',1)[1]
            m=re.search(r'^## '+re.escape(section)+r' .*?(?=^## |\Z)',full,re.M|re.S)
            current=m.group().encode()
        else:current=(ROOT/p['source']).read_bytes()
        slices.append(dict(page_id=ident,source=p['source'],selector=p['selector'],snapshot=p['snapshot'],sha256=sha(ROOT/p['snapshot']),frozen_sha256=p['source_sha256'],current_slice_equals_frozen=current==raw))
    full=(ROOT/'course/first-steps.md').read_text();w1=re.search(r'^## W\.1 .*?(?=^## |\Z)',full,re.M|re.S).group()
    slices.append(dict(page_id='W.1 necessary exact slice',source='course/first-steps.md#W.1',sha256=hashlib.sha256(w1.encode()).hexdigest(),semantics='W.1 installs --extra cpu --group notebook, activates .venv, registers the same kernel. No prerequisite execution or Colab test is claimed here.'))
    base=ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review'
    aliases=load(base/'private-map.json')['aliases']
    grades=load(base/'combined/grades.json')['grades']
    graded={(x['case_id'],aliases[x['candidate']]):x['passed'] for x in grades}
    weights=dict(photo_summary=270,photo_fact=135,text_presence=840,single_ocr=1512,ordered_ocr=5040,text_chat=560,voice_typed_reference_chat=1260,voice_actual_asr_chat=1260)
    variants={};inputs=[]
    for variant in ['base','adapter-step-001039','adapter-step-002077']:
        path=base/'raw'/f'generations-{variant}.json';rows=load(path);counts=Counter();den=Counter();incomplete=[]
        for row in rows:
            g=group(row);key=(row['task']+':'+row['id'],variant)
            if key in graded:passed=graded[key]
            else:
                target=unicodedata.normalize('NFKC',row['reference_answer']).strip()
                answer=unicodedata.normalize('NFKC',row['prediction']).strip()
                if row['task']=='ocr':target=re.sub(r'\s+','',target);answer=re.sub(r'\s+','',answer)
                passed=target==answer
            ok=complete(row)
            if not ok:incomplete.append(row['task']+':'+row['id'])
            counts[g]+=int(ok and passed);den[g]+=1
        variants[variant]=dict(counts=dict(counts),denominators=dict(den),incomplete=len(incomplete),primary_numerator=sum(counts[g]*weights[g] for g in weights),primary_denominator=75600)
        inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    for variant in ['adapter-step-001039','adapter-step-002077']:
        x=variants[variant];before=variants['base']['counts'];after=x['counts'];x['voice_nonregression_pass']={k:after[k]>=before[k] for k in ['voice_typed_reference_chat','voice_actual_asr_chat']}
    inputs += [dict(path=str((base/name).relative_to(ROOT)),sha256=sha(base/name)) for name in ['combined/grades.json','private-map.json','raw/result.json']]
    raw_final=load(ROOT/'docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/result.json')
    model=TinyLM(ModelConfig(max_length=64))
    old_source=[]
    for name in ['scripts/gpu_smoke_check.py','scripts/modal_smoke.py','tiny_perceptron/model.py']:
        raw=subprocess.check_output(['git','show','b75ae446797e138a7ab733ae3c9b02d5ab7fbda4:'+name],cwd=ROOT)
        old_source.append(dict(path=name,revision='b75ae446797e138a7ab733ae3c9b02d5ab7fbda4',sha256=hashlib.sha256(raw).hexdigest(),current_equals_historical=raw==(ROOT/name).read_bytes()))
    journal=next(Path('/workspace/work/reference/omni-technical-validation-v1/github/11405157675').rglob('outputs.jsonl'))
    test_records=[json.loads(line)['record'] for line in journal.read_text().splitlines()]
    ocr_texts=[r['supervision']['ocr_text'] for r in test_records if r['task']=='ocr']
    domains=dict(ocr_characters=sorted(set(''.join(ocr_texts))),ocr_glyph_count=len(set(''.join(ocr_texts))),ocr_lengths=sorted(set(map(len,ocr_texts))),vision_class_ids=sorted({label for r in test_records for label in r.get('supervision',{}).get('vision_labels',[]) if label>=0}),query_slots=sorted({r['supervision']['query_slot'] for r in test_records if r['task']=='vision_relation'}),voice_intents=sorted({r['supervision']['intent'] for r in test_records if r['task']=='voice_qa'}))
    value=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),environment=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),device='CPU',threads='1'),scope='Historical reaggregation from raw decoder records and original per-answer grades. This is not new inference, training, image regrading, or a new final test.',prerequisite_slices=slices,natural_validation=variants,natural_inputs=inputs,v2_raw_test_domains=domains,final_natural_pins={k:raw_final[k] for k in ['model','model_revision','asr_model','asr_revision','total_parameters','split','status']},old_gpu_software_sources=old_source,old_gpu_config_parameter_count=sum(p.numel() for p in model.parameters()),figure_inspection=dict(status='not_applicable',reason='validation snapshot contains no Markdown image, HTML img, SVG, chart, or numerical diagram; its figures_sha256 is {}. Linked chapter diagrams were not used as evidence and no image viewing is claimed.'))
    Path(__file__).with_name('context-recalculation.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(natural_validation=variants,old_gpu_config_parameter_count=value['old_gpu_config_parameter_count'],prerequisite_match=all(x.get('current_slice_equals_frozen',True) for x in slices)),ensure_ascii=False))
