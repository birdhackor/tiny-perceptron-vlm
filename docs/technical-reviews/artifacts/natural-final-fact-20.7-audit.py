from pathlib import Path
import csv,hashlib,json,platform,re,subprocess,sys,unicodedata,importlib.metadata
import soundfile as sf
P=Path('docs/technical-reviews/artifacts/natural-final-fact-20.7-')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def out(n,d): Path(str(P)+n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def distance(a,b):
    # Independent full-matrix Wagner-Fischer recurrence, no project metric import.
    matrix=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): matrix[i][0]=i
    for j in range(len(b)+1): matrix[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1): matrix[i][j]=min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+int(a[i-1]!=b[j-1]))
    return matrix[-1][-1]
def norm(x): return ''.join(unicodedata.normalize('NFKC',x).split())
manifest=json.loads(Path('docs/natural-assistant/manifest.json').read_text())
speech=json.loads(Path('data/natural/speech/manifest.json').read_text())
transcripts=json.loads(Path('docs/natural-assistant/evidence/final/transcripts.json').read_text())
gens=json.loads(Path('docs/natural-assistant/evidence/final/generations-adapter.json').read_text())
allrows=manifest['audio_rows']; rows=[r for r in allrows if r['split']=='test']; assert len(rows)==len(transcripts)==12
metadata={}
for split in ['train','dev','test']:
    p=Path('data/natural/speech/official-sources')/(split+'.tsv')
    declared=speech['sources'][0]['files'][split]['transcripts']; assert digest(p)==declared['sha256']
    fields=[line.split('\t') for line in p.read_text().splitlines()]; assert all(len(x)==7 for x in fields)
    metadata[split]={x[1]:x for x in fields}
records=[]; original42=[]
for row in allrows:
    raw=metadata[row['source_split']][Path(row['audio']).name]
    assert row['user']==raw[2]==row['raw_transcription']
    assert row['normalized_transcription']==raw[3]
    assert row['num_samples']==int(raw[5])
    assert row['answer'] is None and row['speaker'] is None and row['synthetic'] is False
    original42.append({'id':row['id'],'source_split':row['source_split'],'raw_reference_unchanged':True,'assistant_label':None,'sentence_id':raw[0]})
for row,t in zip(rows,transcripts,strict=True):
    assert row['id']==t['id']; assert row['user']==t['reference_transcript']
    ref=t['reference_transcript']; hyp=t['transcript']; raw=distance(ref,hyp); normal=distance(norm(ref),norm(hyp))
    assert raw==t['raw_errors'] and len(ref)==t['raw_reference_characters']; assert normal==t['errors'] and len(norm(ref))==t['reference_characters']
    info=sf.info(Path('data/natural')/row['audio']); audio_hash=digest(Path('data/natural')/row['audio'])
    assert audio_hash==row['sha256']==t['audio_sha256']; assert info.samplerate==16000 and info.frames==row['num_samples']
    assert info.frames/info.samplerate==t['audio_seconds']==row['duration_seconds']; assert info.channels==1
    toks=t['raw_token_ids']; pre=t['expected_decoder_prompt_ids']; suffix=t['generated_token_ids']
    assert toks[:len(pre)]==pre==[50258,50260,50359,50363] and toks[len(pre):]==suffix
    assert len(toks)==t['raw_token_count'] and len(suffix)==t['generated_token_count'] and suffix[-1]==50257
    assert t['eos_token_ids']==[50257] and t['ended_with_eos'] and t['stop_reason']=='eos' and not t['truncated'] and not t['completion_unknown'] and len(suffix)<128
    pair={x['task']:x for x in gens if x['id']==row['id']}
    typed=pair['typed_chat']; spoken=pair['speech_chat']
    assert typed['user']==ref and spoken['user']==hyp and spoken['reference_user']==ref
    for x in [typed,spoken]:
        assert x['image'] is None and x['image_grid_thw'] is None and 'history' not in x
        assert len(x['generated_token_ids'])==x['generated_tokens']; assert x['generated_token_ids'][-1] in x['eos_token_ids']
        assert x['ended_with_eos'] and x['stop_reason']=='eos' and not x['truncated'] and not x['completion_unknown'] and not x['reached_max_new_tokens']
    assert row['history']==[{'role':'system','content':'請用繁體中文簡短回應使用者所說的內容，不要只重複原話。'}]
    records.append({'id':row['id'],'reference':ref,'actual_asr':hyp,'raw_errors':raw,'raw_reference_characters':len(ref),'normalized_errors':normal,'normalized_reference_characters':len(norm(ref)),'normalized_reference':norm(ref),'normalized_prediction':norm(hyp),'audio':row['audio'],'audio_sha256':audio_hash,'sf':{'frames':info.frames,'samplerate':info.samplerate,'channels':info.channels,'seconds':info.frames/info.samplerate,'format':info.format,'subtype':info.subtype},'raw_tokens':len(toks),'generated_tokens':len(suffix),'raw_last_token':toks[-1],'typed_tokens':typed['generated_tokens'],'speech_tokens':spoken['generated_tokens'],'stop_all_three':'eos','history_expected_from_frozen_manifest':row['history'],'actual_serialized_history_saved':False})
total={'raw_errors':sum(r['raw_errors'] for r in records),'raw_reference_characters':sum(r['raw_reference_characters'] for r in records),'normalized_errors':sum(r['normalized_errors'] for r in records),'normalized_reference_characters':sum(r['normalized_reference_characters'] for r in records),'audio_frames':sum(r['sf']['frames'] for r in records),'audio_seconds':sum(r['sf']['seconds'] for r in records),'asr_eos':12,'typed_eos':12,'speech_eos':12}
total.update(raw_cer=total['raw_errors']/total['raw_reference_characters'],normalized_cer=total['normalized_errors']/total['normalized_reference_characters'])
section=re.search(r'^## 20\.7 .+?(?=^## |\Z)',Path('course/chapters/20.md').read_text(),re.M|re.S).group()
code=re.search(r'```python\n(.*?)```',section,re.S).group(1);Path(str(P)+'sum-card.py').write_text(code)
exercise=code.replace('recognized = "請算三加二。"','recognized = "請算十三加二。"').replace('recognized_numbers = (3, 2)','recognized_numbers = (13, 2)')
Path(str(P)+'exercise.py').write_text(exercise)
runs=[]
for n in ['sum-card.py','exercise.py']:
    cmd=[sys.executable,str(P)+n]; r=subprocess.run(cmd,capture_output=True,text=True); assert r.returncode==0
    Path(str(P)+n.replace('.py','.stdout.txt')).write_text(r.stdout);Path(str(P)+n.replace('.py','.stderr.txt')).write_text(r.stderr)
    runs.append({'command':' '.join(cmd),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'scope':'handwritten strings/numeric tuples; genuine Python sum; no ASR, audio playback, LM, parser or model tool call'})
expected=['逐字稿完整符合 False\n原問題的兩數相加 15\n錯誤逐字稿的兩數相加 5\n這是人工漏字示例，沒有辨識錄音或呼叫聊天模型\n','逐字稿完整符合 True\n原問題的兩數相加 15\n錯誤逐字稿的兩數相加 15\n這是人工漏字示例，沒有辨識錄音或呼叫聊天模型\n']
assert [r['stdout'] for r in runs]==expected
unicode_examples={x:{'input_codepoints':[hex(ord(c)) for c in x],'NFKC':unicodedata.normalize('NFKC',x),'normalized_no_whitespace':norm(x)} for x in ['１５','没','沒','Ａ\tＢ\nＣ\u3000Ｄ','，；（）']}
assert norm('没')=='没' and norm('沒')=='沒' and distance(norm('没'),norm('沒'))==1
filehashes={p:digest(p) for p in ['docs/natural-assistant/evidence/final/transcripts.json','docs/natural-assistant/evidence/final/generations-adapter.json','docs/natural-assistant/evidence/final/result.json','docs/natural-assistant/evidence/final/execution.json','docs/natural-assistant/manifest.json','docs/natural-assistant/rubric.json','tiny_perceptron/natural_assistant.py','scripts/natural_assistant.py','scripts/prepare_natural_speech.py','tiny_perceptron/natural_ui.py','data/natural/speech/manifest.json']}
env={'python':platform.python_version(),'soundfile':sf.__version__,'numpy':importlib.metadata.version('numpy'),'torch':importlib.metadata.version('torch'),'unicodedata':unicodedata.unidata_version,'device':'CPU independent text arithmetic/audio metadata inspection; no model inference'}
result={'reviewer_task':'/root/natural_factual_final_20_7','environment':env,'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-final-fact-20.7-audit.py','source_sha256':hashlib.sha256(section.encode()).hexdigest(),'files':filehashes,'original42':original42,'final12':records,'totals':total,'sum_and_exercise':runs,'unicode_examples':unicode_examples,'audio_played':False,'model_rerun':False,'limitation':'Serialized history absent from raw generation records; shared history is verified through same run code, immutable manifest/source hash and paired generation input fields, not replay of a complete logged prompt.'}
out('audit.json',result); print(json.dumps({'environment':env,'section_sha':result['source_sha256'],'original42':len(original42),'totals':total,'sum_and_exercise':runs,'unicode_examples':unicode_examples,'history_limitation':result['limitation']},ensure_ascii=False,indent=2))
