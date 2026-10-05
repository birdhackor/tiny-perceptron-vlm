"""Same original owner: narrow version/metadata reinspection, no repeated experiments."""
from pathlib import Path
import difflib,hashlib,inspect,json,re,sys
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
REL=BASE.relative_to(ROOT).as_posix()
OUT=BASE/'input-semantics-reinspection'
OUT.mkdir(exist_ok=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
history=ROOT/'docs/technical-reviews/history/phase4-5_15-own-before-input-semantics-a4aa33d56b45276e05f682c0ad9aef9a2ed041491f1d28c3fb4878d5dc331592.json'
assert sha(history.read_bytes())=='a4aa33d56b45276e05f682c0ad9aef9a2ed041491f1d28c3fb4878d5dc331592'
report_path=ROOT/'docs/technical-reviews/5.15.json'
report=json.loads(report_path.read_text())
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_5_15'
assert sha(report_path.read_bytes())==sha(history.read_bytes()), 'Do not overwrite an unexpected subsequent report'
oldmeta=json.loads((BASE/'original/extraction.json').read_text())
initial_whole=report.pop('source_file_sha256')
assert initial_whole==oldmeta['source_file_sha256']
raw_chapter=(ROOT/'course/chapters/05.md').read_bytes()
def section(path,lesson):
 raw=path.read_bytes(); heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
 i=next(i for i,h in enumerate(heads) if h[0].startswith(('## '+lesson+' ').encode()))
 return raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)]
read_scope=[]
for lesson,file,snapshot in [('5.15','05.md','original/section.md'),('3.5','03.md','inputs/context-3.5.md'),('5.14','05.md','inputs/context-5.14.md')]:
 current=section(ROOT/'course/chapters'/file,lesson);old=(BASE/snapshot).read_bytes()
 assert current==old, 'Substantive content changed; stop and review actual difference'
 frozen=OUT/('current-'+lesson+'.md');frozen.write_bytes(current)
 read_scope.append({'source':'course/chapters/'+file+'#'+lesson,'current_section_sha256':sha(current),
  'initial_saved_section_path':REL+'/'+snapshot,'initial_saved_section_sha256':sha(old),
  'current_saved_section_path':frozen.relative_to(ROOT).as_posix(),'bytes_equal':True,
  'actually_read':'Latest full section text personally read in this turn; byte equality also independently computed.'})
method_path=ROOT/'docs/review-tools/factual-reviewer-instructions.md'
method=method_path.read_bytes();(OUT/'factual-reviewer-instructions.md').write_bytes(method)
code_versions=[]
for name in ['docs/review-tools/section_facts.py','scripts/build_course.py','scripts/check_technical_reviews.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','tiny_perceptron/data.py','tiny_perceptron/training.py']:
 current=(ROOT/name).read_bytes();old=(BASE/'inputs'/name).read_bytes()
 assert current==old, 'Execution dependency changed; inspect before reusing results'
 code_versions.append({'path':name,'current_sha256':sha(current),'saved_initial_sha256':sha(old),'bytes_equal':True})
oldproject=(BASE/'inputs/pyproject.toml').read_bytes();currentproject=(ROOT/'pyproject.toml').read_bytes()
patch=''.join(difflib.unified_diff(oldproject.decode().splitlines(True),currentproject.decode().splitlines(True),fromfile='frozen pyproject.toml',tofile='current pyproject.toml'))
(OUT/'pyproject.diff').write_text(patch)
(OUT/'current-pyproject.toml').write_bytes(currentproject)
assert len([line for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')])==2
assert '+    "docs/course-revision-20261005",' in patch
assert not any(line.startswith('-') and not line.startswith('---') for line in patch.splitlines())
import torch
env={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),
 'torch_git_version':torch.version.git_version,'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu'}
oldenv=json.loads((BASE/'installed-environment.json').read_text())
for key in ['python','torch','torch_git_version','cuda_build','cuda_available','device']:
 assert env[key]==oldenv[key], 'Environment changed; do not reuse blindly'
contracts=[]
for name,obj in [('Embedding',torch.nn.Embedding),('manual_seed',torch.manual_seed),('normal_',torch.nn.init.normal_),('_manual_seed_impl',torch.random._manual_seed_impl),('_no_grad_normal_',torch.nn.init._no_grad_normal_),('Linear_reset_parameters',torch.nn.Linear.reset_parameters),('LayerNorm_reset_parameters',torch.nn.LayerNorm.reset_parameters)]:
 current=inspect.getsource(obj).encode();old=(BASE/'installed'/(name+'.py')).read_bytes()
 assert current==old
 contracts.append({'name':name,'current_sha256':sha(current),'initial_sha256':sha(old),'bytes_equal':True})
for name,obj in [('Tensor-item',torch.Tensor.item),('torch-std',torch.std),('torch-equal',torch.equal)]:
 current=obj.__doc__.encode();old=(BASE/'installed'/(name+'.txt')).read_bytes()
 assert current==old
 contracts.append({'name':name,'current_sha256':sha(current),'initial_sha256':sha(old),'bytes_equal':True})
verified_artifacts=[]
for artifact in report['artifacts']:
 current=sha((ROOT/artifact['path']).read_bytes())
 assert current==artifact['sha256'], 'Initial artifact changed: '+artifact['id']
 verified_artifacts.append({'id':artifact['id'],'path':artifact['path'],'sha256':current,'matches_original_report':True})
fence=(BASE/'original/fence-1.py').read_bytes()
body=(OUT/'current-5.15.md').read_bytes()
assert fence in body
assert oldmeta['python_fences'][0]['sha256']==sha(fence)
assert report['figure_sha256']=={} and oldmeta['svg_references']==[]
assert not re.findall(rb'!\[[^\]]*\]\(([^)]+\.svg)\)',body)
receipt={
 'schema_version':1,'kind':'same_owner_narrow_input_semantics_reinspection','lesson_id':'5.15',
 'reviewer_task':report['reviewer_task'],'reviewed_on':'2026-10-05',
 'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-5_15-independent/reinspect_input_semantics.py',
 'shell':'bash login:false','initial_report_history':{'path':history.relative_to(ROOT).as_posix(),'sha256':sha(history.read_bytes()),'preservation':'Opaque byte copy before reading own report metadata; never modified.'},
 'latest_method':{'path':method_path.relative_to(ROOT).as_posix(),'sha256':sha(method),'snapshot':(OUT/'factual-reviewer-instructions.md').relative_to(ROOT).as_posix(),'actual_read':'Entire latest method personally read before reinspection.'},
 'actual_read_scope':read_scope,
 'complete_markdown_initial_input_semantics':{'source':'course/chapters/05.md','initial_recorded_sha256':initial_whole,
  'provenance_path':REL+'/original/extraction.json','provenance_sha256':sha((BASE/'original/extraction.json').read_bytes()),
  'how_recorded':'Initial section_facts.py original_section reads the whole Markdown bytes to extract 5.15, then stores SHA-256 of those bytes.',
  'initial_whole_markdown_bytes_retained':False,
  'meaning':'Historical acquisition identity only; does not mean reviewer read the whole chapter, does not claim complete old chapter snapshot exists, and is not the current whole chapter version.'},
 'current_complete_markdown_observation':{'source':'course/chapters/05.md','sha256':sha(raw_chapter),
  'scope':'Current file identity at narrow reinspection, not a whole-chapter technical verdict.'},
 'execution_dependencies':code_versions,
 'pyproject_delta':{'initial_sha256':sha(oldproject),'current_sha256':sha(currentproject),
  'diff_path':(OUT/'pyproject.diff').relative_to(ROOT).as_posix(),
  'personally_read_delta':'Only Ruff extend-exclude adds a comment and docs/course-revision-20261005 directory; no dependency or initialization semantics changed.'},
 'current_environment':env,'installed_contracts':contracts,
 'exact_original_fence_sha256':sha(fence),'figure_sha256':{},'figure_result':'No original/current SVG reference; no new visual scope or browser check claimed.',
 'reused_original_artifacts':verified_artifacts,
 'reused_evidence_scope':'All original CPU original-fence, exact no-reset variation, Decimal arithmetic, RNG-data-stream probes and personally inspected official snapshots remain unchanged. This turn checks exact identities and scope only; no fresh CPU experiment or official-source download is claimed.',
 'result':'pass maintained for the unchanged 5.15 and necessary context; removed ambiguous top-level full-file hash and made retained section/context vs historical acquisition identity explicit.',
 'unresolved_substantive_dependencies':[],
 'limitation':'Complete initial chapter raw bytes were not saved; only its genuine acquisition fingerprint remains in unchanged original extraction metadata. Actual initial reviewed section and necessary context bytes are retained and equal to current bytes.'}
receipt_path=OUT/'receipt.json';save(receipt_path,receipt)
def artifact(identifier,path,kind,description):
 report['artifacts'].append({'id':identifier,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.read_bytes()),'kind':kind,'description':description})
artifact('input_semantics_reinspection',receipt_path,'execution','原技術owner本輪真實窄版本複查receipt；明確復用原證據，非重做CPU。')
report['artifacts'][-1].update(command=receipt['command'],result=receipt['result'],environment=env)
artifact('input_semantics_reinspection_code',Path(__file__),'code','本轮same-owner exact bytes/hash comparison與metadata更新實作。')
artifact('latest_factual_method',OUT/'factual-reviewer-instructions.md','source_snapshot','本人本輪親讀的最新完整方法。')
artifact('current_dependency_delta',OUT/'pyproject.diff','source_snapshot','唯一相鄰設定變更：Ruff exclude；無執行语义影響。')
for item in read_scope:
 artifact('current_scope_'+item['source'].split('#')[-1].replace('.','_'),ROOT/item['current_saved_section_path'],'source_snapshot','窄複查本人親讀的目前小節/必要前文原bytes。')
report['frozen_input']={'meaning':'Retained bytes actually reviewed in the original 5.15 review, plus their historical acquisition metadata.',
 'source':'course/chapters/05.md#5.15','section_path':REL+'/original/section.md','section_sha256':report['source_sha256'],
 'extraction_metadata_path':REL+'/original/extraction.json','extraction_metadata_sha256':sha((BASE/'original/extraction.json').read_bytes()),
 'historical_complete_markdown_input_record':receipt['complete_markdown_initial_input_semantics'],
 'necessary_context':[{'source':i['source'],'path':i['initial_saved_section_path'],'sha256':i['initial_saved_section_sha256']} for i in read_scope if not i['source'].endswith('#5.15')]}
report['current_input_observation']={'checked_scope':[{'source':i['source'],'sha256':i['current_section_sha256']} for i in read_scope],
 'complete_markdown_file_sha256_at_reinspection':sha(raw_chapter),'whole_file_meaning':'Current file fingerprint observation only; verdict remains scoped to 5.15 and genuine prerequisites.'}
report['same_owner_reinspection']={'artifact_id':'input_semantics_reinspection','path':receipt_path.relative_to(ROOT).as_posix(),'sha256':sha(receipt_path.read_bytes()),
 'history_path':history.relative_to(ROOT).as_posix(),'history_sha256':sha(history.read_bytes()),'result':'pass','experiment_evidence':'Reused exact unchanged originals; no new CPU or source fetch','unresolved_substantive_dependencies':[]}
report['checks']['source_verification']['details']+=' SAME owner窄複查確認本節/必要ctx/實作/安裝契約與47原引用artifact精確未變；來源全檔599ad1…改為歷史取得身份，当前整章不冒充同版本。'
save(report_path,report)
print(json.dumps({'report':report_path.relative_to(ROOT).as_posix(),'report_sha256':sha(report_path.read_bytes()),
 'history':history.relative_to(ROOT).as_posix(),'history_sha256':sha(history.read_bytes()),
 'receipt':receipt_path.relative_to(ROOT).as_posix(),'receipt_sha256':sha(receipt_path.read_bytes()),
 'source_sha256':report['source_sha256'],'figure_sha256':{},'original_artifacts_reused':len(verified_artifacts),
 'current_whole_sha256':sha(raw_chapter),'initial_whole_sha256_historical_only':initial_whole,
 'verdict':report['verdict'],'unresolved_substantive_dependencies':[]},ensure_ascii=False,indent=2))
