"""Render already available, hash-verified photos and save named raw leaves."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
INPUT = OUT / 'inputs'
manifest = json.loads((INPUT / 'docs/natural-assistant/v4/manifest.json').read_text())
rows = [r for r in manifest['rows'] if r['split']=='validation' and r['task']=='scene']
photos = [r for r in rows if r['id'].endswith('/scene')]
vroot = INPUT / 'outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review'
variants = ['base','adapter-step-001039','adapter-step-002077']
records = {v: json.loads((vroot / ('generations-'+v+'.json')).read_text()) for v in variants}
indices = {v: {r['task']+':'+r['id']: (i,r) for i,r in enumerate(records[v])} for v in variants}
grades = json.loads((INPUT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/combined/grades.json').read_text())['grades']
grade_index = {(g['case_id'],g['candidate']):(i,g) for i,g in enumerate(grades)}
aliases = {'base':'B','adapter-step-001039':'A','adapter-step-002077':'C'}
render_receipt = []
for batch in range(7):
    subset=photos[batch*4:(batch+1)*4]
    canvas=Image.new('RGB',(1400,1050),'white')
    draw=ImageDraw.Draw(canvas)
    info=[]
    for j,p in enumerate(subset):
        img=Image.open(OUT/'source-photos'/Path(p['image']).name)
        img.thumbnail((690,480))
        x=(j%2)*700; y=(j//2)*525
        canvas.paste(img,(x+(700-img.width)//2,y))
        draw.text((x+10,y+485),str(batch*4+j+1)+' '+Path(p['image']).name,fill='black')
        tasks=[]
        for row in rows:
            if row['image']==p['image']:
                entry={'id':row['id'],'user':row['user'],'gold_example':row['answer'],'rubric':row['references']['rubric'],'outputs':{}}
                for v in variants[:2]:
                    _,raw=indices[v][row['task']+':'+row['id']]
                    _,g=grade_index[(row['task']+':'+row['id'],aliases[v])]
                    entry['outputs'][v]={'prediction':raw['prediction'],'passed':g['passed'],'reason':g['reason']}
                tasks.append(entry)
        info.append({'photo':Path(p['image']).name,'tasks':tasks})
    path=OUT/f'photo-sheet-{batch+1}.png'
    before=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    canvas.save(path)
    after=hashlib.sha256(path.read_bytes()).hexdigest()
    assert before is None or before==after, 'Previously viewed pixels changed'
    (OUT/f'photo-inspection-input-{batch+1}.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    render_receipt.append({'path':path.name,'sha256':after,'size':[1400,1050], 'unchanged_from_viewed_render':before==after})
voice=[]
for i,row in enumerate(manifest['audio_rows']):
    if row['split']=='validation' and row['task']=='speech_chat':
        data={'manifest_pointer':f'/audio_rows/{i}','id':row['id'],'reference_user':row['user'],'system':row.get('system'),'rubric':row['references']['rubric'],'outputs':[]}
        for v in variants:
            for task in ['typed_chat','speech_chat']:
                ri,raw=indices[v][task+':'+row['id']]
                gi,grade=grade_index[(task+':'+row['id'],aliases[v])]
                data['outputs'].append({'variant':v,'task':task,'generation_pointer':f'/{ri}',
                    'grade_pointer':f'/grades/{gi}','user':raw['user'],'prediction':raw['prediction'],
                    'passed':grade['passed'],'reason':grade['reason']})
        voice.append(data)
(OUT/'voice-inspection-input.json').write_text(json.dumps(voice,ensure_ascii=False,indent=2)+'\n')
(OUT/'render-receipt.json').write_text(json.dumps(render_receipt,indent=2)+'\n')
print(json.dumps({'source_photos':len(photos),'rendered_sheets':len(render_receipt),'photo_response_pairs':len(rows)*2,'voice_questions':len(voice),'voice_responses':sum(len(x['outputs']) for x in voice),'previously_viewed_pixels_unchanged':all(x['unchanged_from_viewed_render'] for x in render_receipt)}))
