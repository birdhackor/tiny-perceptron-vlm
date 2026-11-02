"""Exact96 source and original-receipt guards in private fixtures; no provider calls."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts.omni_revision import research_execution as runner

ROOT=Path(__file__).resolve().parents[5]
SNAPSHOT=Path('/workspace/work/omni-P5-full-tools-review-source-v6')
QWEN=Path('/workspace/work/omni-qwen25-original-attempt-41015365689')
controls=[]


def reject(name,function):
    try:
        function()
    except ValueError as error:
        controls.append({'case':name,'rejected':True,'error':str(error)})
    else:
        raise AssertionError('accepted invalid '+name)


names=runner.source_paths()
assert len(names)==96
audit=runner.AUDIT.relative_to(ROOT)
with tempfile.TemporaryDirectory(prefix='p5-current96-source-')as directory:
    private=Path(directory)
    for name in [*names,str(audit)]:
        target=private/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SNAPSHOT/name,target)
    def git(*args):
        return subprocess.check_output(['git','-C',str(private),*args],stderr=subprocess.STDOUT)
    git('init','-q')
    git('add','.')
    git('-c','user.name=Private fixture','-c','user.email=fixture@example.invalid','commit','-qm','Exact96 private source fixture')
    revision=git('rev-parse','HEAD').decode().strip()
    with patch.object(runner,'ROOT',private),patch.object(runner,'FROZEN',private/'docs/omni-products/v1'),patch.object(runner,'AUDIT',private/audit),patch.dict(os.environ,{'GITHUB_SHA':revision}):
        source=runner.checked_sources()
        assert len(source['files'])==96
        controls.append({'case':'exact96_all_actual_private_Git_objects_positive','accepted':True})
        for name in names:
            target=private/name
            original=target.read_bytes()
            target.write_bytes(original+b'\n')
            reject('byte_drift:'+name,runner.checked_sources)
            target.write_bytes(original)
        target=private/audit
        original=target.read_bytes()
        target.write_bytes(original+b'\n')
        reject('audit_uncommitted_even_when_semantic_JSON_unchanged',runner.checked_sources)
        target.write_bytes(original)
        with patch.dict(os.environ,{'GITHUB_SHA':'f'*40}):
            reject('Actions_SHA_not_actual_checkout',runner.checked_sources)
        assert runner.checked_sources()==source

# Use actual preserved Qwen records as the positive, not fabricated success.
sources=json.loads((QWEN/'artifact/sources.json').read_bytes())
reserved=json.loads((QWEN/'artifact/reservation.json').read_bytes())['attempt']
receipt=json.loads((QWEN/'artifact/research-receipt.json').read_bytes())
attempt=reserved['attempt_id']
runner.check_receipt_identity('qwen25',attempt,sources,reserved,receipt)
controls.append({'case':'actual_original_failed_Qwen_identity_positive','accepted':True,'not_a_capability_pass':True})
for key in ['schema','attempt','job','source_revision','controller_audit_sha256','release_tag','release_body','input_payload_sha256','config_sha256','core_selection_sha256','core_final_opening_sha256','preparation_recovery']:
    bad=copy.deepcopy(receipt)
    bad[key]='foreign'
    reject('foreign_receipt_'+key,lambda bad=bad:runner.check_receipt_identity('qwen25',attempt,sources,reserved,bad))
for key in ['sources_sha256','controller_audit_sha256','ownership','input_sha256','job_config_sha256','core_selection_sha256','core_final_opening_sha256']:
    bad=copy.deepcopy(reserved)
    purpose=json.loads(bad['purpose'])
    purpose[key]='foreign'
    bad['purpose']=json.dumps(purpose)
    reject('foreign_reserved_purpose_'+key,lambda bad=bad:runner.check_receipt_identity('qwen25',attempt,sources,bad,receipt))
provider_calls=[]
def forbidden_provider(*args,**kwargs):
    provider_calls.append(True)
    raise AssertionError('provider access forbidden')
with tempfile.TemporaryDirectory(prefix='p5-foreign-mature-receipt-')as directory:
    output=Path(directory)
    (output/'reservation.json').write_text(json.dumps({'attempt':reserved}))
    bad=copy.deepcopy(receipt)
    bad.update(schema='omni-cpu-core-execution-v1',attempt='cpu-core-foreign',success=True)
    (output/'research-receipt.json').write_text(json.dumps(bad))
    with patch.object(runner,'identity',lambda *args:None),patch.object(runner,'checked_sources',lambda:sources),patch.object(runner,'api',forbidden_provider),patch.object(runner,'GitHubLedger',forbidden_provider),patch.object(runner,'bound',forbidden_provider):
        reject('foreign_true_core_shape_rejected_before_bound_provider_or_writer',lambda:runner.finish_run('qwen25',attempt,output))
assert provider_calls==[]
out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),'source_count':96,'source_drift_cases':96,'control_count':len(controls),'controls':controls,'all_expected':True,'provider_calls':0,'main_repository_commits':0,'private_Git_fixture_only':True,'new_permit_or_terminal_job_retry':False,'NN_or_training':False,'limits':'True checked_sources/committed code operates against actual private Git objects; receipt positive is original failed Qwen. All mutations are private memory/files. No run/reserve/Modal/workflow/publication calls.'}
path=Path(__file__).with_name('scope450-mature-admission-controls-result-v1.json')
with path.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_expected':True,'source_count':96,'source_drift_cases':96,'total_controls':len(controls),'provider_calls':0},ensure_ascii=False))
