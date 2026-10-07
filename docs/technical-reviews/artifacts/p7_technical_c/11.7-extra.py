import pathlib
print('half recipe',{'文字有效位置':500/1000,'圖文有效位置':500/1000},'recorded unchanged',recorded)
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vqa-raw.json').read_text())['results']
for n in ['projector_only','partial','all','all_replay']:
 v=r['variants'][n];before=v['text_before']['samples'];after=v['text_after']['samples'];assert [s['messages'] for s in before]==[s['messages'] for s in after]
 print('history',n,'before',sum(s['generated']==s['expected'] for s in before),len(before),'after',sum(s['generated']==s['expected'] for s in after),len(after),'newly correct',sum(not a['exact'] and b['exact'] for a,b in zip(before,after)),'newly wrong',sum(a['exact'] and not b['exact'] for a,b in zip(before,after)),'image',sum(s['generated']==s['target'] for s in v['test']['samples']),len(v['test']['samples']),'targets',v['training']['effective_targets'],'prob',v['replay_probability_per_example'])
