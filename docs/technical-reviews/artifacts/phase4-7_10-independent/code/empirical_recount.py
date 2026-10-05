"""Recount existing UltraChat evidence, independent of a training run."""
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sha = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
raw_bytes = (BASE/'inputs/train-first-100.jsonl').read_bytes()
assert sha(raw_bytes) == '70d360fbb3c30482cd337c69c927c00644f565b604a38c5a82a33535988dd1b7'
raw = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
r = json.loads((BASE/'inputs/docs/course-experiments/results/sft.json').read_text())
pilot = r['results']['ultrachat_pilot']
records = []
trimmed = {'user':0,'assistant':0}
for index, row in enumerate(raw):
    user = next((m['content'] for m in row['messages'] if m['role']=='user'),None)
    assistant = next((m['content'] for m in row['messages'] if m['role']=='assistant'),None)
    if user and assistant:
        cropped = {role:text.encode()[:120].decode('utf-8',errors='ignore') for role,text in [('user',user),('assistant',assistant)]}
        for role, text in [('user',user),('assistant',assistant)]:
            trimmed[role] += int(cropped[role] != text)
            assert len(cropped[role].encode()) <= 120
            assert text.startswith(cropped[role])
        records.append({'family':row.get('prompt_id',sha(canonical(user))),'source_row':index,'source_record_sha256':sha(canonical(row)),'scope':'first-turn UTF-8-safe excerpt','messages':[{'role':'user','content':cropped['user']},{'role':'assistant','content':cropped['assistant']}]})
assert len(raw) == len(records) == 100
groups={}
for row in records: groups.setdefault(str(row['family']),[]).append(row)
assert len(groups)==100
keys=sorted(groups); random.Random(42).shuffle(keys)
parts={split:[row for key in selected for row in groups[key]] for split,selected in [('train',keys[:80]),('validation',keys[80:90]),('test',keys[90:])]}
stats={}
for split, rows in parts.items():
    content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
    assert sha(content)==pilot['data'][split]['sha256']
    # Each two-message chat has X/Y length user_bytes + answer_bytes + 4.
    # The assistant role predicts the first answer byte; EOS adds one target.
    lengths=[len(row['messages'][0]['content'].encode())+len(row['messages'][1]['content'].encode())+4 for row in rows]
    answer_counts=[len(row['messages'][1]['content'].encode())+1 for row in rows]
    assert max(lengths)<=256 and min(answer_counts)>0
    stats[split]={'records':len(rows),'families':len({row['family'] for row in rows}),'jsonl_sha256':sha(content),'max_X_Y_positions':max(lengths),'min_supervised_tokens':min(answer_counts),'supervised_tokens_once_through_split':sum(answer_counts)}
    if split!='train':
        report=pilot['evaluation'][split]
        assert len(report['samples'])==report['records']==report['examples']==len(rows)
        assert sum(answer_counts)==report['effective_tokens']
        matches=0; ended=0
        previews=[]
        for row,sample in zip(rows,report['samples'],strict=True):
            assert sample['messages']==row['messages'][:-1]
            assert sample['expected']==row['messages'][-1]['content']
            ids=sample['generated_ids']
            before_eos=ids[:ids.index(2)] if 2 in ids else ids
            target_ids=[b+8 for b in row['messages'][-1]['content'].encode()]
            exact=before_eos==target_ids
            assert exact==sample['exact']
            decoded=bytes(v-8 for v in before_eos if v>=8).decode('utf-8',errors='replace')
            assert decoded==sample['generated']
            matches+=int(exact);ended+=int(2 in ids)
            previews.append({'source_row':row['source_row'],'expected_prefix':sample['expected'][:80],'generated_prefix':decoded[:100],'generated_tokens':len(ids),'exact':exact,'eos':2 in ids})
        assert matches==report['matches']==0 and report['exact_match']==0
        assert abs(ended/len(rows)-report['eos_rate'])<1e-12
        assert abs(report['nll_sum']/sum(answer_counts)-report['nll'])<1e-12
        stats[split].update(matches=matches,exact_match=matches/len(rows),EOS_count=ended,nll_sum=report['nll_sum'],nll_nats=report['nll'],samples=previews)
train=pilot['training']
assert train['records']==80 and train['steps']==80 and train['history'][-1]['step']==80
assert sha(json.dumps(parts['train'],sort_keys=True,ensure_ascii=False).encode())==train['records_sha256']
sampler=random.Random(42)
exposure=0
for step in range(80):
    batch=sampler.choices(parts['train'],k=4)
    exposure+=sum(len(row['messages'][-1]['content'].encode())+1 for row in batch)
assert exposure==train['effective_tokens']==38714
for a in parts:
 for b in parts:
  if a!=b: assert not ({row['family'] for row in parts[a]} & {row['family'] for row in parts[b]})
result={'source':'Existing docs/course-experiments/results/sft.json, not retraining','revision':r['revision'],'raw_records':len(raw),'prepared_records':len(records),'UTF8_cropped_records':trimmed,'split_family_overlap':0,'splits':stats,'training':{'updates':train['steps'],'batch_size':4,'sampled_example_exposures':80*4,'effective_token_exposures_recomputed':exposure,'train_once_tokens':stats['train']['supervised_tokens_once_through_split'],'history_steps':[entry['step'] for entry in train['history']],'logged_training_device':r['device'],'verification_device':'CPU, no model loaded'},'tolerance':'IDs/counts/hashes exact; NLL division and EOS rates 1e-12 absolute'}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
