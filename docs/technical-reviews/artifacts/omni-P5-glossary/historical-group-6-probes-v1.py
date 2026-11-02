"""Issue-specific historical callback after full-current receipt; no model work."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.posttraining import ppo_clipped_objective

PROOF=Path(__file__).resolve().parent
OUTSIDE=Path('/workspace/work/reference/omni-P5-glossary-whole')
TASK='/root/omni_p5_technical_glossary_whole'
REQUEST='docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json'
REQUEST_SHA='714f16f7d5c20924e20f91a8b48bca7a87bd77e2602d1a1ae501d06d93f819e5'
FULL='docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/technical-current-v1.json'
FULL_SHA='8974f763d5fab6c4ad6c67beff96039cf941c688a88d38b801e3d16a47ab253a'
ORIGINAL='docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/technical-original-v1.json'
ORIGINAL_SHA='1e4367497e8dec241dfc0cf142cfe886a2ba2d1dac376ed623a7e6dcc7aece1a'
INTAKE='docs/course-revision-20261029-omni/reviews/P5-actual/technical-original-collection-intake-v15.json'
INTAKE_SHA='535b46de6b4da1c137163098a617db3330b0457c86cbaf8c35128c4673b15c80'


def reference(path):
    raw=(ROOT/path).read_bytes()
    return {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def read(reference):
    assert globals()['reference'](reference['path'])['sha256']==reference['sha256']
    return json.loads((ROOT/reference['path']).read_bytes())


def pointer(document,path):
    for token in path.split('/')[1:]:
        token=token.replace('~1','/').replace('~0','~')
        document=document[int(token)] if isinstance(document,list) else document[token]
    return document


def small_excerpt(text,term,extent=650):
    position=text.index(term)
    return text[max(0,position-50):position+extent]


def main():
    started=datetime.now(timezone.utc).isoformat()
    requests=read({'path':REQUEST,'sha256':REQUEST_SHA})
    group=requests['groups'][6]
    assert group['page_id']=='glossary' and group['old_owner']=='/root/rv1019_technical_glossary'
    assert len(group['items'])==4
    current=read({'path':FULL,'sha256':FULL_SHA})
    original=read({'path':ORIGINAL,'sha256':ORIGINAL_SHA})
    intake=read({'path':INTAKE,'sha256':INTAKE_SHA})
    receipts=[(i,r) for i,r in enumerate(intake['collection_index']) if r.get('path')==FULL and r.get('sha256')==FULL_SHA]
    assert len(receipts)==1
    index,receipt=receipts[0]
    at=datetime.fromisoformat
    assert at(original['created_at'])<=at(current['created_at'])<=at(receipt['received_at'])<at(started)
    assert receipt['reviewer_task']==TASK and receipt['collection_role']=='technical'
    freeze=read(current['canonical_source_freeze'])
    page=next(p for p in freeze['pages'] if p['page_id']=='glossary')
    raw=(ROOT/page['snapshot']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==current['source_sha256']==original['source_sha256']
    assert raw==(ROOT/'course/glossary.md').read_bytes()
    text=raw.decode()
    unicode_line=next(line for line in text.splitlines() if line.startswith('| 字元、碼點'))
    ppo_line=next(line for line in text.splitlines() if line.startswith('| 近端策略最佳化'))
    assert '少數字元由幾個碼點組成（例如帶膚色的 👍🏽）' in unicode_line
    assert '每個碼點（U+D800–U+DFFF 除外）存成 1 到 4 個 byte' in unicode_line
    assert '設法避免一次把回答的機率改變太多，但不保證每次的改變都很小' in ppo_line
    histories=[]
    for item in group['items']:
        issue=pointer(read(item['original_report']),item['json_pointer'])
        disposition=pointer(read(item['disposition_ref']),item['disposition_ref']['json_pointer'])
        verification=pointer(read(item['historical_fix_verification']['report']),item['historical_fix_verification']['json_pointer'])
        assert item['necessity']==disposition['final_necessity']=='necessary'
        assert disposition['action']=='fixed' and disposition['issue_id']==item['issue_id']
        assert disposition['report']==item['original_report'] and disposition['json_pointer']==item['json_pointer']
        assert item['historical_fix_verification'] in disposition['verification']
        assert verification['verdict']=='pass' and verification['lesson_id']=='glossary'
        histories.append({'issue_id':item['issue_id'],'disposition_ref':item['disposition_ref'],
                          'original_report':item['original_report'],'json_pointer':item['json_pointer'],
                          'historical_fix_verification':item['historical_fix_verification'],
                          'original_issue_status':issue['status'],'original_issue_necessary':issue['necessary'],
                          'necessity':item['necessity'],'action':disposition['action'],
                          'affected_pages':disposition['affected_pages']})
    unavailable=pointer(read(group['unavailability']['report']),group['unavailability']['json_pointer'])
    assert unavailable['reviewer_task']==group['old_owner'] and unavailable['unavailable'] is True
    assert any(e['tool']=='collaboration.followup_task' and 'not found' in e['response'] for e in unavailable['events'])
    assert any(e['tool']=='collaboration.list_agents' and e['response'].get('agents')==[] for e in unavailable['events'])
    samples={}
    for value in ['👍🏽','e\u0301','🇺🇳','👨‍👩‍👧‍👦']:
        samples[value]={'codepoints':[f'U+{ord(c):04X}' for c in value],
                        'codepoint_count':len(value),'utf8_bytes':list(value.encode('utf-8')),
                        'total_bytes':len(value.encode('utf-8')),
                        'bytes_per_codepoint':[len(c.encode('utf-8')) for c in value]}
    assert samples['👍🏽']['codepoint_count']==2 and samples['👍🏽']['total_bytes']==8
    assert samples['e\u0301']['codepoint_count']==2 and samples['🇺🇳']['codepoint_count']==2
    assert samples['👨‍👩‍👧‍👦']['codepoint_count']==7
    rejected=[]
    for value in [0xD800,0xDFFF]:
        try:chr(value).encode('utf-8')
        except UnicodeEncodeError:rejected.append(f'U+{value:04X}')
    assert rejected==['U+D800','U+DFFF']
    old_log=torch.tensor([math.log(.5)],dtype=torch.float64)
    new_log=torch.tensor([math.log(.9)],dtype=torch.float64)
    advantage=torch.tensor([1.],dtype=torch.float64)
    result=ppo_clipped_objective(new_log,old_log,advantage,clip_range=.2)
    assert abs(result['ratio'].item()-1.8)<1e-12
    assert abs(result['clipped'].item()-1.2)<1e-12 and abs(result['surrogate'].item()-1.2)<1e-12
    assert abs(new_log.exp().item()-.9)<1e-12
    unicode_original=(OUTSIDE/'unicode.txt').read_bytes()
    utf8_original=(OUTSIDE/'utf8.txt').read_bytes()
    guide_original=(OUTSIDE/'historical-ppo-guide.html').read_bytes()
    primary={s['id']:s for s in current['sources'] if s['id'] in ['s-unicode','s-utf8','s-ppo']}
    assert hashlib.sha256(unicode_original).hexdigest()==primary['s-unicode']['raw_sha256']
    assert hashlib.sha256(utf8_original).hexdigest()==primary['s-utf8']['raw_sha256']
    utext=(OUTSIDE/'unicode.plain').read_text()
    ptext=(OUTSIDE/'historical-ppo-guide.txt').read_text()
    excerpts={'unicode_section3':small_excerpt(utext,'A single Unicode code point is often, but not always',800),
              'utf8_surrogate':small_excerpt(utf8_original.decode(),'The definition of UTF-8 prohibits',450),
              'ppo_background':small_excerpt(ptext,'PPO-Clip doesn’t have a KL-divergence term',600),
              'ppo_caveat':small_excerpt(ptext,'it is still possible to end up with a new policy',650)}
    evidence={'started_at':started,'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':TASK,
              'kind':'historical-group6-issue-specific-evidence-after-full-current-intake',
              'command':'uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-glossary/historical-group-6-probes-v1.py',
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU'},'status':'passed',
              'request':reference(REQUEST),'group_index':6,'full_review':reference(FULL),'own_original':reference(ORIGINAL),
              'actual_full_review_intake':{'report':reference(INTAKE),'json_pointer':f'/collection_index/{index}','record':receipt},
              'canonical_source_freeze':current['canonical_source_freeze'],'current_source_sha256':current['source_sha256'],
              'current_figure_sha256':{},'current_quotes':{'unicode':unicode_line,'ppo':ppo_line},
              'historical_instances':histories,'unavailability':group['unavailability'],
              'unavailability_record_preserved':{'reviewer_task':unavailable['reviewer_task'],'recorded_at':unavailable['recorded_at'],
                                                'reason':unavailable['reason'],'events':unavailable['events']},
              'unicode_scalar_examples':samples,'strict_utf8_surrogates_rejected':rejected,
              'ppo_scalar_example':{'old_probability':.5,'new_probability':.9,'advantage':1.,'epsilon':.2,
                                     'actual_ratio':result['ratio'].item(),'surrogate_clipped_term':result['clipped'].item(),
                                     'surrogate_min':result['surrogate'].item(),'actual_probability_after_objective':new_log.exp().item(),
                                     'no_optimizer_or_parameter_update':True},
              'current_posttraining_source':reference('tiny_perceptron/posttraining.py'),
              'small_primary_excerpts':excerpts,
              'new_ppo_guide_original':{'url':'https://spinningup.openai.com/en/latest/algorithms/ppo.html','raw_bytes':len(guide_original),
                                        'raw_sha256':hashlib.sha256(guide_original).hexdigest(),'full_text_storage':'outside Git'},
              'scope':'Four supplied historical instances only; two issue families. Both own independent original and received current are immutable. Scalar examples and direct source inspection only; no model inference, training, policy tuning, workflow or seal.'}
    path=PROOF/'historical-group-6-probes-v1.json';assert not path.exists();path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'instances':len(histories)},ensure_ascii=False))


if __name__=='__main__':main()
