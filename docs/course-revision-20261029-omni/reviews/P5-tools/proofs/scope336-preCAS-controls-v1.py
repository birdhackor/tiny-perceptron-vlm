"""Download/preflight rejection must stop before pricing, reservation and CAS."""
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts.omni_revision import cpu_final_execution as runner

ENV = {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':runner.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40}
results=[]
for failure in ('download', 'loader'):
    with tempfile.TemporaryDirectory(prefix='p5-before-CAS-') as directory:
        root=Path(directory)
        events=[]
        class Ledger:
            def read(self):
                events.append('ledger_read_only')
                return {'attempts':[]}, 'observed'
            def write(self, *args):
                raise AssertionError('no CAS before successful preparation')
        def child(arguments, log, seconds, state):
            name=arguments[1]
            events.append(name)
            log.write_text('synthetic no NN child\n')
            state['all_training_processes_exited']=True
            if name.endswith('legacy_download'):
                assert seconds==900
                if failure=='download':
                    raise RuntimeError('expected-download-failure')
            else:
                assert name.endswith('core_final') and arguments[2]=='legacy-preflight' and seconds==180
                raise RuntimeError('expected-loader-failure')
        def forbidden(*args, **kwargs):
            raise AssertionError('pricing/reservation/CAS not reachable')
        with patch.dict(os.environ, ENV), patch.object(runner,'ROOT',root), patch.object(runner,'checked_sources',lambda:{'revision':'a'*40}), patch.object(runner,'GitHubLedger',Ledger), patch.object(runner,'prepare_core',lambda *args:events.append('load_only_core')), patch.object(runner.transport,'bounded_child',child), patch.object(runner.transport,'pricing',forbidden), patch.object(runner,'reserve',forbidden):
            try:
                runner.reserve_final('cpu-final-123-1',root/'control')
            except RuntimeError as error:
                assert str(error)=='expected-'+failure+'-failure'
            else:
                raise AssertionError('failed preparation accepted')
        assert events==['ledger_read_only','load_only_core','scripts.omni_revision.legacy_download']+(['scripts.omni_revision.core_final'] if failure=='loader' else [])
        assert not (root/'control/reservation.json').exists()
        assert not (root/'control/final-run-started.json').exists()
        results.append({'case':failure+'_failure','events':events,'no_pricing_reservation_CAS_or_final_start':True,'original_log_retained':True})
for filename in ('legacy-download.log','legacy-preflight.log','legacy-download.json','legacy-preflight.json'):
    with tempfile.TemporaryDirectory(prefix='p5-legacy-preserve-') as directory:
        output=Path(directory)
        original=b'preserve previous original bytes\n'
        (output/filename).write_bytes(original)
        with patch.object(runner.transport,'bounded_child',side_effect=AssertionError('previous evidence must reject before child')):
            try:
                runner.legacy_inputs(output)
            except FileExistsError:
                pass
            else:
                raise AssertionError('previous evidence overwritten')
        assert (output/filename).read_bytes()==original
        results.append({'case':filename+'_preserved','rejected_before_child':True,'original_bytes_unchanged':True})
out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),'controls':results,'control_count':len(results),'all_expected':True,'no_real_network_ledger_or_NN':True,'limits':'Explicit private child failures and in-memory ledger; no production reservation, provider query, upload, paid work or final measurement.'}
path=Path(__file__).with_name('scope336-preCAS-controls-result-v1.json')
with path.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
