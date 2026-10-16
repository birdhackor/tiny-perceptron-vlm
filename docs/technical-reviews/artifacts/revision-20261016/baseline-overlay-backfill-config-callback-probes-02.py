"""知情配置補件：只加原CSS、保留pyproject原版；真舊validator與負向replay。"""
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).parent
BATCH='docs/course-revision-20261016/'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(name):return {'path':name,'sha256':sha(ROOT/name)}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
 start=datetime.now(UTC).isoformat()
 overlay_name=BATCH+'baseline-overlay.json';before_name=BATCH+'baseline-overlay-before-css-backfill-01.json';css_name=BATCH+'baseline-overlay/course/web/course.css';config_name=BATCH+'baseline-overlay/pyproject.toml'
 overlay=json.loads((ROOT/overlay_name).read_bytes());prior_raw=(ROOT/before_name).read_bytes();prior=json.loads(prior_raw)
 previous_report=BATCH+'reviews/baseline-overlay-config-callback-01.json';previous_raw=(ROOT/previous_report).read_bytes();previous=json.loads(previous_raw)
 assert hashlib.sha256(previous_raw).hexdigest()=='4d98e991bd10d37299cc9492b5ffbca2abacbc167da38b0d28d4f9af2e400ad1'
 prior_overlay=next(x for x in previous['checked_files'] if x['path']==overlay_name)
 assert sha(ROOT/before_name)==prior_overlay['sha256']=='ea7e4ef127ba6e470bbc9cbdbcb5d87df3e877992f7c7db07e9fa3c3257a4877'
 remainder=copy.deepcopy(overlay);addition=remainder['files'].pop();assert remainder==prior and len(overlay['files'])==9
 assert overlay['baseline_revision']==prior['baseline_revision']=='2a550a44386b046543c4ed08f50bdaf5280213e7'
 assert addition=={'live_path':'course/web/course.css',**ref(css_name)}
 assert overlay['files'][7]=={'live_path':'pyproject.toml',**ref(config_name)}
 assert sha(ROOT/config_name)=='8285022edba54024f998970b62759151d8c7edcbfc0d04aa83e47fa864c1b850'
 checked_names=[overlay_name,before_name,css_name,config_name,'pyproject.toml','course/web/course.css',previous_report,BATCH+'all-gate-12.json',BATCH+'backfill-author-baseline-css-observation-01.json','docs/course-publish-20261008/review-manifest.json','docs/course-publish-20261008/current-source-freeze.json','docs/course-repair-20261008/reviews/freeze-01/manifest.json','docs/course-repair-20261008/reviews/freeze-01/checks/technical-inputs-manifest.json','docs/review-tools/composite_review.py','docs/review-tools/baseline_replay.py','tests/test_composite_course_review.py','tests/test_incremental_revision_review.py',BATCH+'reviews/tool-audit-callback-04.json']
 before=[ref(name) for name in checked_names]
 verified=[]
 for item in overlay['files']:
  assert sha(ROOT/item['path'])==item['sha256'];verified.append(item)
 git_checks=[]
 for live,snapshot in [('course/web/course.css',css_name),('pyproject.toml',config_name)]:
  raw=subprocess.check_output(['git','show',overlay['baseline_revision']+':'+live],cwd=ROOT)
  assert raw==(ROOT/snapshot).read_bytes();git_checks.append({'live_path':live,'snapshot':snapshot,'sha256':sha(ROOT/snapshot),'byte_exact_baseline_git':True})
 manifest=json.loads((ROOT/'docs/course-publish-20261008/review-manifest.json').read_bytes())
 catalog=json.loads((ROOT/manifest['catalog']['path']).read_bytes());frozen_ref=catalog['manifest_references']['freeze01'];frozen=json.loads((ROOT/frozen_ref['path']).read_bytes())
 assert sha(ROOT/frozen_ref['path'])==frozen_ref['sha256']
 assert frozen['implementation_sha256']['course/web/course.css']==addition['sha256']=='f1ce0d9b952328c4def76f83f85ae020a524bb6155c63b0130bd82c64e63ee9a'
 assert sha(ROOT/'course/web/course.css')!=addition['sha256']
 techref=manifest['current_technical_inputs'];assert sha(ROOT/techref['path'])==techref['sha256'];tech=json.loads((ROOT/techref['path']).read_bytes())
 assert 'course/web/course.css' not in tech['files_sha256']
 # Name precise historical channel: CSS belongs to implementation map, not the147-file technical-input map.
 tools=json.loads((ROOT/(BATCH+'reviews/tool-audit-callback-04.json')).read_bytes());toolmap={x['path']:x['sha256'] for x in tools['checked_files']}
 core=['docs/review-tools/incremental_revision.py','docs/review-tools/baseline_replay.py','docs/review-tools/phase7_review.py',BATCH+'tools/reader.py',BATCH+'tools/freeze_sources.py','tests/test_incremental_revision_review.py','docs/review-tools/composite_review.py']
 for name in core:assert sha(ROOT/name)==toolmap[name]
 composite=load('untouched_composite_config02','docs/review-tools/composite_review.py')
 class PinnedConfigStore:
  def raw(self,reference):
   assert reference==manifest['configuration_delta']['baseline_snapshot']
   raw=(ROOT/reference['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==reference['sha256'];return raw
 store=PinnedConfigStore();baseline_digest=manifest['configuration_delta']['baseline_snapshot']['sha256']
 config_error=None
 try:composite.check_configuration_delta(store,manifest,baseline_digest,(ROOT/'pyproject.toml').read_bytes())
 except Exception as error:config_error=str(error)
 assert config_error=='configuration delta does not bind current bytes'
 composite.check_configuration_delta(store,manifest,baseline_digest,(ROOT/config_name).read_bytes())
 replay=load('untouched_baseline_config02','docs/review-tools/baseline_replay.py')
 baseline_report_name='docs/technical-reviews/artifacts/revision-20261016/baseline-overlay-backfill-config-callback-replay-02.json'
 baseline=json.loads((ROOT/baseline_report_name).read_bytes());assert baseline['status']=='passed_metadata_checks' and not baseline['failures'] and len(baseline['records'])==325
 original_read=Path.read_bytes;hits=[]
 def old_overlay_read(path):
  if path==ROOT/overlay_name:hits.append(str(path));return prior_raw
  return original_read(path)
 # Actual helper + actual old validator, only replacing its overlay input with preserved original8 bytes.
 with patch.object(Path,'read_bytes',new=old_overlay_read):missing_css=replay.replay(ROOT)
 assert hits and missing_css['status']=='failed' and missing_css['failures']==['current implementation drift: course/web/course.css']
 control=ART/'baseline-overlay-missing-css-control-02.json'
 with control.open('x') as stream:json.dump(missing_css,stream,ensure_ascii=False,indent=2);stream.write('\n')
 failure=json.loads((ROOT/(BATCH+'all-gate-12.json')).read_bytes());assert failure['status']=='failed' and failure['failures']==["preserved baseline failed: ['current implementation drift: course/web/course.css']"]
 assert before==[ref(name) for name in checked_names]
 report={'reviewer_task':'/root/tech_20261016_review_tools','initial_context':'informed','current_context':'informed','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'torch_distribution':importlib.metadata.version('torch')},'checked_files':before,'verified_overlay_entries':verified,'only_overlay_addition':addition,'old8_overlay_byte_identity':ref(before_name),'git_snapshot_identity':git_checks,'old_css_identity_channel':{'implementation_map':{'path':frozen_ref['path'],'sha256':frozen_ref['sha256'],'css_sha256':frozen['implementation_sha256']['course/web/course.css']},'technical_inputs':{**techref,'css_not_in_this_map':True,'files_count':len(tech['files_sha256'])}},'configuration_control':{'current_bytes':'rejected','error':config_error,'restored_original_snapshot':'accepted'},'core_bytes_unchanged_from_tool_callback04':[ref(name) for name in core],'baseline_replay':{**ref(baseline_report_name),'status':baseline['status'],'record_count':325,'failures':baseline['failures']},'missing_css_negative_control':{**ref(control.relative_to(ROOT).as_posix()),'status':missing_css['status'],'failures':missing_css['failures'],'input_only_override':'Original preserved8 overlay bytes; validator/helper/source evidence not changed.'},'preserved_failure':ref(BATCH+'all-gate-12.json'),'all_expected':True,'limits':['Only informed overlay/configuration closure and actual old325 metadata replay; no allgate12 success, fresh reader, scientific/physical-isolation/human comprehension proof or new mobileCSS technical review.','Did not modify implementation/criteria/tests/current CSS/source/seals/old reports. Negative control overrides one read input in own process only, no source mutation.']}
 path=ART/'baseline-overlay-backfill-config-callback-probes-results-02.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'all_expected':True,'only_addition':addition['live_path'],'positive_records':325,'missing_css_control':missing_css['failures'],'config_controls':['current rejected','original accepted'],'sha256':sha(path)}))


if __name__=='__main__':main()
