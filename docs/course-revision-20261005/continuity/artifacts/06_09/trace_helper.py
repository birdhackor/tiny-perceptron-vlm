import json, pathlib, hashlib, datetime, sys, re
ROOT = pathlib.Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-revision-20261005/continuity'
INV = {p['page_id']:p for p in json.loads((BASE/'inventory.json').read_text())['pages']}
TRACE = BASE/'traces/06_09.jsonl'
def append(page, section, previous, new_question, switching, understanding, doubts=None, extra=None):
    p=INV[page]
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reviewer_task':'/root/continuity_06_09','fork_turns':'none','event':'initial_read','page_id':page,'section':section,'source':p['source'],'snapshot':p['snapshot'],'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],'previous_actually_read':previous,'new_question_and_need':new_question,'switching':switching,'current_understanding':understanding,'current_doubts':doubts or [],'extra_inference':[],'missing_figures':[]}
    if extra:row.update(extra)
    with TRACE.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def read(page):
    p=INV[page]; content=(ROOT/p['snapshot']).read_text(); actual=hashlib.sha256(content.encode()).hexdigest()
    assert actual==p['source_sha256'],(page,actual)
    print('PAGE',page,'SOURCE SHA256',actual,'FIGURES',json.dumps(p['figures_sha256'],ensure_ascii=False)); print(content)
if __name__=='__main__': read(sys.argv[1])
