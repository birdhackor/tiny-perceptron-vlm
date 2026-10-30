"""Additional CancelledError controls using the disclosed author transport fixture, actual staged code."""
import asyncio,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'original-source';sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('mature_transport_fixture',SOURCE/'tests/test_omni_mature_research.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
root=BASE/'probes/original-staged-cancellation';root.mkdir(parents=True,exist_ok=False);original=fixture.staged.run_staged_job;cases=[]
for mode in ('cancel_create_response_without_ID','cancel_task_ID_after_known_sandbox'):
 output=root/mode;output.mkdir();captured=[]
 async def run(*args):
  if mode=='cancel_task_ID_after_known_sandbox':
   hydrate=args[3]._new_hydrated
   def altered(*values):
    sandbox=hydrate(*values)
    async def cancel(**kwargs):raise asyncio.CancelledError('explicit review-only task-ID cancellation')
    sandbox._get_task_id=cancel;return sandbox
   args[3]._new_hydrated=altered
  value=await original(*args);captured.append(value);return value
 with patch.object(fixture.staged,'run_staged_job',side_effect=run),patch.object(fixture,'TimeoutError',asyncio.CancelledError,create=True):fixture.test_single_create_staged_transfer_and_exact_stop(output,'lost_empty' if mode=='cancel_create_response_without_ID' else 'worker_failure')
 value=captured[-1];assert value['error_type']=='CancelledError' and not value['success']
 expected=mode=='cancel_task_ID_after_known_sandbox';assert value['cleanup_checks']['owned_sandboxes_stopped'] is expected
 cases.append({'case':mode,'error_type':value['error_type'],'success':value['success'],'paid_call_started':value['paid_call_started'],'sandbox_id':value.get('sandbox_id'),'known_and_discovered_IDs':value.get('observed_owned_sandbox_ids'),'stopped_sandboxes':value.get('stopped_sandboxes'),'cleanup_checks':value['cleanup_checks']})
record={'scope':'Disclosed reuse of author SimpleNamespace transport fixture with added CancelledError injection. Real staged source and cleanup run, no SDK client or remote/model/final calls.','cases':cases,'real_remote_or_neural_or_final_calls':0}
p=BASE/'original-staged-cancellation-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
