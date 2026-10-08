import json, sys, subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent
n=json.load(sys.stdin)
for a,k in [('p','page_id'),('i','unit_index'),('u','understanding'),('m','materials_and_labels'),('e','expected_change'),('c','confusion_and_quote'),('v','missing_visuals')]:
 if a in n:n[k]=n.pop(a)
if 'q' in n:
 q=n.pop('q');n['four_questions']={'q1':q[0],'q2_mechanism':{'answer':q[1],'status':'已足夠' if not q[1].startswith('待教') else '合理待教'},'q2_need':{'answer':q[2],'status':'已足夠' if not q[2].startswith('待教') else '合理待教'},'q3':q[3],'q4':q[4]}
n['reviewer']='/root/read_foundations'
n['route']='grouped sequential main-only; no optional or external materials'
p=B/'notes'/f"foundations-{n['page_id']}-{n['unit_index']:03d}.json"
p.write_text(json.dumps(n,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(B/'reader.py'),'next','foundations','--note',str(p)],check=True)
