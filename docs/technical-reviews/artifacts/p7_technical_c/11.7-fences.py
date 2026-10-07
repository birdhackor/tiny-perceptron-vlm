import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
recipe = {"文字有效位置": 750, "圖文有效位置": 250}
recorded = {
    "訓練前": {"文字答對": 8, "圖文答對": 2},
    "訓練後": {"文字答對": 6, "圖文答對": 7},
}
total_positions = sum(recipe.values())
ratios = {}
for category, positions in recipe.items():
    ratios[category] = positions / total_positions
print("示例訓練比例", ratios)
for phase, scores in recorded.items():
    print(phase, "文字", scores["文字答對"] / 10, "圖文", scores["圖文答對"] / 10)


# Owner supplied proportional check
import pathlib
print('half recipe',{'文字有效位置':500/1000,'圖文有效位置':500/1000},'recorded unchanged',recorded)
r=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/vqa-raw.json').read_text())['results']
for n in ['projector_only','partial','all','all_replay']:
 v=r['variants'][n];before=v['text_before']['samples'];after=v['text_after']['samples'];assert [s['messages'] for s in before]==[s['messages'] for s in after]
 print('history',n,'before',sum(s['generated']==s['expected'] for s in before),len(before),'after',sum(s['generated']==s['expected'] for s in after),len(after),'newly correct',sum(not a['exact'] and b['exact'] for a,b in zip(before,after)),'newly wrong',sum(a['exact'] and not b['exact'] for a,b in zip(before,after)),'image',sum(s['generated']==s['target'] for s in v['test']['samples']),len(v['test']['samples']),'targets',v['training']['effective_targets'],'prob',v['replay_probability_per_example'])

