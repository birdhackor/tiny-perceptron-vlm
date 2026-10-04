"""CPU tokenizer/source integrity checks; no model or processor inference."""
import hashlib,json,sys,urllib.request
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
token_path=ROOT/'outputs/natural-extension/research/ocr/qwen3-vl-pinned-tokenizer.json'
tok=Tokenizer.from_file(str(token_path))
token_url='https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/tokenizer.json'
with urllib.request.urlopen(token_url,timeout=45) as r:official=r.read()
assert hashlib.sha256(official).hexdigest()==sha(token_path)
generation_url='https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/generation_config.json'
with urllib.request.urlopen(generation_url,timeout=45) as r:genraw=r.read()
(OUT/'natural-final-fact-20.6-qwen-generation-config.json').write_bytes(genraw)
genconfig=json.loads(genraw)
receipt={'python':sys.version,'device':'CPU','tokenizer_url':token_url,'tokenizer_sha256':sha(token_path),'generation_config_url':generation_url,'generation_config_sha256':hashlib.sha256(genraw).hexdigest(),'token_checks':[],'input_hashes':{},'execution_checks':[]}
for group in ['final','validation','external-ocr']:
    combined=read(f'docs/natural-assistant/evidence/{group}/generations.json')
    for variant in ['base','adapter']:
        path=f'docs/natural-assistant/evidence/{group}/generations-{variant}.json'
        data=read(path)
        receipt['input_hashes'][path]=sha(ROOT/path)
        combined_sub=[r for r in combined if r['variant']==variant]
        assert data==combined_sub
        for r in data:
            if r['task'] not in ['ocr','text_presence']:continue
            assert tok.decode(r['generated_token_ids'],skip_special_tokens=True)==r['prediction']
            assert r['eos_token_ids']==genconfig['eos_token_id']
            receipt['token_checks'].append({'group':group,'id':r['id'],'variant':variant,'decoded_exactly':True,'length':len(r['generated_token_ids']),'last_token_id':r['generated_token_ids'][-1],'EOS':r['ended_with_eos'],'tokens_sha256':hashlib.sha256(json.dumps(r['generated_token_ids']).encode()).hexdigest()})

selection=read('docs/natural-assistant/selection.json')
for group in ['final','external-ocr']:
    path=f'docs/natural-assistant/evidence/{group}/execution.json'
    e=read(path)
    assert sha(ROOT/'docs/natural-assistant/selection.json')==e['selection_sha256']
    assert e['pretest_selection']['adapter_sha256']==selection['adapter_sha256']==e['adapter_sha256']
    assert e['pretest_selection']['validation_result_sha256']==selection['validation_result_sha256']
    assert e['runner_arguments'][e['runner_arguments'].index('--max-pixels')+1]=='524288'
    assert e['runner_arguments'][e['runner_arguments'].index('--max-new-tokens')+1]=='384'
    receipt['execution_checks'].append({'path':path,'sha256':sha(ROOT/path),'run_id':e['run_id'],'revision':e['revision'],'selected_before_test_sha256':e['selection_sha256'],'adapter_sha256':e['adapter_sha256'],'max_pixels':524288,'max_new_tokens':384})

ext=read('docs/natural-assistant/external-ocr.json');m=read('docs/natural-assistant/manifest.json')
assert not {r['family'] for r in ext['rows']} & {r['family'] for r in m['rows']}
assert ext['training_allowed'] is False and all(r['split']=='test' for r in ext['rows'])
receipt['external_course_split_excluded']=True
ocr=read('outputs/natural-extension/data/ocr/manifest.json')
images={im['image']:im for im in ocr['images']}
receipt['final_order_layouts']=[{'id':r['id'],'answer':r['answer'],'layout':images[r['image']]['layout']} for r in ocr['rows'] if r['split']=='test' and r['references']['kind']=='ocr_order']
g=read('docs/natural-assistant/evidence/external-ocr/generations.json')
rx=[r for r in g if r['id']=='cc-ocr-scene-101' and r['variant']=='base'][0]
assert rx['prediction'].splitlines()==['万民药房']*77
rx=[r for r in g if r['id']=='cc-ocr-vertical-000' and r['variant']=='adapter'][0]
assert rx['prediction'].splitlines()==['细节展示','纯原创设计']+['ABCD']*126
receipt['repetition_checks']={'base_pharmacy_lines':77,'base_pharmacy_nonwhitespace_characters':308,'base_pharmacy_raw_characters':384,'adapter_dense_ABCD_lines':126,'adapter_dense_total_lines':128}
receipt['percentages']={f'{a}/{b}':round(a/b*100,2) for a,b in [(1,190),(1187,484),(729,484),(953,431),(567,431),(1,10)]}
receipt['input_hashes'].update({p:sha(ROOT/p) for p in ['docs/natural-assistant/manifest.json','docs/natural-assistant/external-ocr.json','docs/natural-assistant/DATA.md','docs/natural-assistant/rubric.json','docs/natural-assistant/selection.json','scripts/prepare_natural_ocr.py','tiny_perceptron/natural_assistant.py']})
(OUT/'natural-final-fact-20.6-recheck-supplement.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'tokens_checked':len(receipt['token_checks']),'tokenizer_sha256':receipt['tokenizer_sha256'],'execution_checks':receipt['execution_checks'],'final_order_layouts':receipt['final_order_layouts'],'repetition_checks':receipt['repetition_checks'],'percentages':receipt['percentages'],'external_course_split_excluded':True},ensure_ascii=False,indent=2))
