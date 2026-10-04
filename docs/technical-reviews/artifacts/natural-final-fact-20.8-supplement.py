import hashlib,json,pathlib,datetime
from scripts.check_technical_reviews import FRONT_MATTER, sections
R=pathlib.Path.cwd();A=R/'docs/technical-reviews/artifacts';E=R/'docs/natural-assistant';hashes={}
def read(p):
 p=R/p;hashes[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
m=read('docs/natural-assistant/manifest.json');pub=read('docs/natural-assistant/public-release.json');rubric=read('docs/natural-assistant/rubric.json');sem=read('docs/natural-assistant/evidence/semantic-final/final-semantic-review.json');gen=read('docs/natural-assistant/evidence/final/generations-adapter.json');exe=read('docs/natural-assistant/evidence/final/execution.json');external=read('docs/natural-assistant/evidence/external-ocr/execution.json')
assert rubric==sem['rubric_unchanged'] and hashes['docs/natural-assistant/rubric.json']==sem['rubric_sha256']
assert hashes['docs/natural-assistant/manifest.json']==pub['manifest_sha256']==rubric['dataset_manifest_sha256']==sem['selected_adapter_and_selection_lock']['manifest_sha256']
sha=next(f['sha256'] for f in pub['files'] if f['output']=='adapter_model.safetensors')
assert sha==exe['adapter_sha256']==external['adapter_sha256']==sem['selected_adapter_and_selection_lock']['adapter_sha256']
assert exe['selection_sha256']==external['selection_sha256']==sem['selected_adapter_and_selection_lock']['selection_sha256']
rows={x['id']:x for x in m['rows']};n=0
for g in gen:
 if g['task'] not in ('speech_chat','typed_chat'):
  r=rows[g['id']];assert g['user']==r['user'] and g['image']==r.get('image') and g['reference_answer']==r['answer'];n+=1
assert n==66
chapters=sorted(int(p.stem) for p in (R/'course/chapters').glob('*.md') if p.stem.isdigit());nbs=list((R/'notebooks').rglob('*.ipynb'));files=sorted((R/'course/chapters').glob('*.md'))+[R/'course'/p for p in FRONT_MATTER];ids=[i for p in files for i,_ in sections(p)]
assert len(chapters)==20 and len(nbs)==261 and len(ids)==287
for p in ['README.md','README_en.md','docs/natural-assistant/STUDENT.md']:
 hashes[p]=hashlib.sha256((R/p).read_bytes()).hexdigest()
body=(R/'course/chapters/20.md').read_text().split('## 20.8 ',1)[1];body='## 20.8 '+body;h=hashlib.sha256(body.encode()).hexdigest();assert h=='3733ef68cd74b299286ad7c19ee6603ab71c6fc3add030b15931e238e458fa91'
r={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':'.venv-natural/bin/python '+str(pathlib.Path(__file__).relative_to(R)),'input_sha256':hashes,'rubric_and_dataset_fixed_hash_match':True,'all66_non_speech_original_prompts_images_GTs_equal_manifest':True,'original_pretest_selected_adapter_SHA_preserved':sha,'selection_SHA_preserved':exe['selection_sha256'],'student_checkout_commit':'1d5fccbc76a6ebbb3757f9b31df17a7c22ea171b','public_HF_provenance_source_commit':pub['git_revision'],'source_provenance_not_student_checkout':True,'current_lesson_sha256':h,'optional_readme_delta_counts':{'numeric_chapters':20,'notebooks':261,'numbered_sections':287},'scope':'Supplemental immutable rubric/GT/selection and README count comparisons; no GT edit/model reselect/forward/GPU/installation/Git mutation.'}
(A/'natural-final-fact-20.8-supplement.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
