from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,difflib,platform
R=Path('/workspace/work/omni-book-integrated-v1')
O=Path('/tmp/c-v3-v21-canonical-readme')
def h(b):return hashlib.sha256(b).hexdigest()
old=R/'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v2/sources/readme.md';new=R/'README.md'
p=R/'docs/course-experiments/public-models.json';d=json.loads(p.read_text());counts=[{'id':x['id'],'weight_files':sum(f['path'].endswith(('.pt','.safetensors')) for f in x['files'])} for x in d['models']]
oldtext=old.read_text();newtext=new.read_text()
needles={
 'v2_product_fact_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('[第19章整合專題]')),
 'legacy_b_c_product_fact_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('Thinking effort')),
 'installation_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('先安裝 [uv]')),
 'historical_platform_scope_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('各版本的顯卡')),
 'dry_run_explanation':next(x for x in oldtext.split('\n\n') if x.startswith('先做一次通路檢查')),
 'dry_run_expected_values':next(x for x in oldtext.split('\n\n') if x.startswith('確認 `mode`')),
 'v2_public_delivery_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('想先觀察第19章')),
 'v2_data_package_paragraph':next(x for x in oldtext.split('\n\n') if x.startswith('要自己從隨機初始化重做')),
 'old_public_model_runtime_example':next(x for x in oldtext.split('\n\n') if x.startswith('第一條只列清單')),
 'retained_omni_av_sound_metrics':'影音任務問「聲音響起時畫面是圓還是方」：每個情境只移動聲音形成兩題，兩題皆答對才算一個配對成功，事前選定模型為200／200。有限音型生成則按指定頻率、次序、時值與停頓判整段音型，200／200通過；換波形內容的影音診斷仍有失敗。',
 'retained_c_v2_metrics':'60題答對39題；固定規則接受其中53題，裡面19題答錯，未達升版門檻'
}
identity={k:{'old_contains':v in oldtext,'current_contains':v in newtext,'substring_sha256':h(v.encode()),'bytes':len(v.encode())} for k,v in needles.items()}
assert all(x['old_contains'] and x['current_contains'] for x in identity.values())
assert len(counts)==30 and sum(x['weight_files'] for x in counts)==120
r={'created_at':datetime.now(timezone.utc).isoformat(),'actor':'/root/c_v3_technical_coord/canonical_readme','environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu; text/JSON only'},'scope':'Independent source-byte delta and original manifest counts; no remote download or retraining. Identity proves retention eligibility, not the truth of old empirical measurements.','source18_readme_sha256':h(old.read_bytes()),'current_readme_sha256':h(new.read_bytes()),'identical_retained_claim_text':identity,'source_delta':list(difflib.unified_diff(oldtext.splitlines(keepends=True),newtext.splitlines(keepends=True),fromfile=str(old.relative_to(R)),tofile='README.md')),'public_model_manifest_path':str(p.relative_to(R)),'public_model_manifest_sha256':h(p.read_bytes()),'public_model_groups':len(counts),'per_group_weight_counts':counts,'total_weight_files':sum(x['weight_files'] for x in counts),'scope_note':'Weight entries count excludes LICENSE/README/manifests/tokenizer/evaluation metadata. No successful remote transport inferred from fixed URLs or SHA lists.'}
q=O/'source-delta-and-manifests.json';q.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(q),'bytes':q.stat().st_size,'sha256':h(q.read_bytes()),'all_retained_substrings_present':True,'model_groups':len(counts),'weight_files':r['total_weight_files']},ensure_ascii=False))
