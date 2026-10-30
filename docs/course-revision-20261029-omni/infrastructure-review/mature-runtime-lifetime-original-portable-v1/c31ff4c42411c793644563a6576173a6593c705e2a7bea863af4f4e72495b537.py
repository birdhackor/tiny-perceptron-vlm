"""真 Modal1.6.0 弱參照生命週期；GC控制不替換filesystem方法，不連服務。"""

import asyncio,gc,hashlib,importlib.metadata,inspect,json,platform,socket,sys,weakref
from datetime import UTC,datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
import modal
from modal._utils.async_utils import synchronizer
from modal.sandbox import _Sandbox
from modal.sandbox_fs import _SandboxFilesystem
from modal_proto import api_pb2
from scripts.omni_revision.execution import hard_wait

assert modal.__version__=='1.6.0' and sys.version_info[:2]==(3,13)
events=[];refs=[];connects=[]
def no_network(*a,**k):connects.append(True);raise AssertionError('no network')
def make():
 sandbox=_Sandbox._new_hydrated('sb-lifetimefixture',SimpleNamespace(),api_pb2.SandboxHandleMetadata(app_id='ap-lifetimefixture'))
 refs.append(weakref.ref(sandbox));return sandbox
async def blocked_exec(*a,**k):
 events.append('exec-entered')
 try:await asyncio.Future()
 finally:events.append('exec-cancelled')
async def main():
 cases=[]
 for operation in ('write','read','inverse'):
  fs=make().filesystem
  owner_ref=refs[-1]
  gc.collect()
  assert owner_ref() is None,'temporary owner must be collectable'
  before=len(events)
  coroutine=fs.write_bytes(data=b'fixture',remote_path='/tmp/input') if operation=='write' else fs.read_bytes('/tmp/input') if operation=='read' else fs.write_bytes('/tmp/input',b'fixture')
  try:await hard_wait(coroutine,.01)
  except (ReferenceError,TypeError) as error:kind=type(error).__name__;message=str(error)
  else:raise AssertionError('collected-owner case unexpectedly succeeded')
  assert kind==('TypeError' if operation=='inverse' else 'ReferenceError')
  assert len(events)==before
  cases.append({'case':'temporary_owner_'+operation,'owner_collected_after_gc':True,'error_type':kind,'message':message,'exec_boundary_calls':0})
 for operation in ('write','read'):
  sandbox=make();fs=sandbox.filesystem;owner_ref=refs[-1]
  gc.collect();assert owner_ref() is sandbox
  before=len(events)
  pending=fs.write_bytes(data=b'fixture',remote_path='/tmp/input') if operation=='write' else fs.read_bytes('/tmp/input')
  try:await hard_wait(pending,.01)
  except TimeoutError:pass
  else:raise AssertionError('blocked operation must time out')
  await asyncio.sleep(0)
  assert events[before:]==['exec-entered','exec-cancelled'] and owner_ref() is sandbox
  cases.append({'case':'strong_owner_'+operation,'owner_survives_gc_and_timeout':True,'error_type':'TimeoutError','exec_boundary_calls':1,'exec_cancel_observed':True})
  del pending,fs,sandbox
  for _ in range(3):await asyncio.sleep(0)
  gc.collect();assert owner_ref() is None
 return cases
with patch.object(_Sandbox,'exec',new=blocked_exec),patch.object(socket.socket,'connect',new=no_network),patch.object(socket,'create_connection',new=no_network):
 cases=synchronizer.create_blocking(main)()
assert not connects
source=Path(inspect.getfile(_SandboxFilesystem));raw=source.read_bytes();dep=HERE/'dependency-snapshot';dep.mkdir(exist_ok=True);(dep/'sandbox_fs.py').write_bytes(raw)
result={'schema':'omni-mature-runtime-lifetime-real-sdk-original-probe-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'python':platform.python_version(),'modal_version':modal.__version__,'synchronicity_version':importlib.metadata.version('synchronicity'),'gc_enabled':gc.isenabled(),'gc_controls':'gc.collect with actual weakrefs; never disable GC or replace filesystem methods','cases':cases,'sdk_filesystem_source':{'path':str(source),'snapshot_path':str(dep/'sandbox_fs.py'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'short_supporting_excerpt':'self._container = cast(_Sandbox | _SidecarContainer, weakref.proxy(container))'},'real_rpc_calls':0,'nn_predictions':0,'network_connect_attempts':connects,'limitations':['GC-forced deterministic reproduction is a private SDK fixture, not proof every unmodified pytest run fails. Actual clean original pytest results are retained separately.','Only exec/RPC boundary is mocked; _new_hydrated, filesystem property, weakref proxy, write/read bodies and hard_wait are real.']}
(HERE/'lifetime-probe-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'python':platform.python_version(),'cases':len(cases),'collected_owner_write_read':'ReferenceError before fake exec','strong_owner_write_read':'TimeoutError + real coroutine cancellation','real_rpc_calls':0}))
