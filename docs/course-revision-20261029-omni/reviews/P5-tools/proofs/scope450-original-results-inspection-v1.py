"""Original package/row/receipt recomputation only; no neural predictions or paid calls."""
import copy
import hashlib
import json
import math
import pickle
import socket
import stat
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import torch

from scripts.omni_revision import core_evaluation as measured
from scripts.omni_revision import core_report, finance, research_execution, research_measurements

ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path('/workspace/work/omni-P5-full-tools-review-source-v6')
MODELS = Path('/workspace/work/omni-models-v1')
CORE = Path('/workspace/work/omni-core-completed-40988311100')
FINAL = Path('/workspace/work/omni-final-completed-41004229262')
KEV = Path('/workspace/work/omni-kev-recovery-completed-41011677439')
QWEN = Path('/workspace/work/omni-qwen25-original-attempt-41015365689')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_archive(path, digest, manifest_name=None):
    assert sha(path) == digest
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = {member.filename for member in members}
        assert len(names) == len(members) <= 1024
        assert sum(member.file_size for member in members) <= 256*1024**2
        for member in members:
            name = PurePosixPath(member.filename)
            assert not name.is_absolute() and '..' not in name.parts and '\\' not in member.filename
            assert not stat.S_ISLNK(member.external_attr >> 16)
        payloads = 0
        if manifest_name:
            bindings = json.loads(archive.read(manifest_name))
            assert names == set(bindings) | {manifest_name}
            for name, binding in bindings.items():
                raw = archive.read(name)
                assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
                payloads += 1
        else:
            # Reading each member also verifies CRC, without executing contents.
            for member in members:
                archive.read(member.filename)
                payloads += 1
        return {'sha256':digest,'archive_bytes':path.stat().st_size,'entries':len(members),'payloads_verified':payloads,'manifest_sha256':hashlib.sha256(archive.read(manifest_name)).hexdigest() if manifest_name else None}


float_differences = []


def equal_or_small_float(actual, expected, pointer=''):
    if type(actual) is not type(expected):
        assert type(actual) in (int,float) and type(expected) in (int,float), pointer
    if isinstance(actual,dict):
        assert actual.keys() == expected.keys(), pointer
        for key in actual:
            equal_or_small_float(actual[key],expected[key],pointer+'/'+str(key))
    elif isinstance(actual,list):
        assert len(actual)==len(expected), pointer
        for index,(left,right) in enumerate(zip(actual,expected)):
            equal_or_small_float(left,right,pointer+'/'+str(index))
    elif type(actual) in (float,int) and type(expected) in (float,int):
        assert math.isfinite(actual) and math.isfinite(expected), pointer
        if actual != expected:
            assert type(actual) is float and type(expected) is float, pointer
            delta=abs(actual-expected)
            assert delta <= 1e-11, (pointer,actual,expected)
            float_differences.append({'pointer':pointer,'absolute_difference':delta})
    else:
        assert actual==expected, (pointer,actual,expected)


attempts = []


def no_forward(module,*args,**kwargs):
    attempts.append(type(module).__name__)
    raise AssertionError('review forbids model calls')


def forbidden(*args,**kwargs):
    raise AssertionError('review forbids network or pickle/model continuation loading')


with patch.object(torch.nn.Module,'_call_impl',no_forward), patch.object(torch,'load',forbidden), patch.object(pickle,'load',forbidden), patch.object(pickle,'loads',forbidden), patch.object(socket,'getaddrinfo',forbidden), patch.object(socket.socket,'connect',forbidden):
    try:
        torch.nn.Identity()(torch.tensor(0.0))
    except AssertionError:
        pass
    assert attempts==['Identity']
    attempts.clear()
    core_archive=checked_archive(CORE/'core-checkpoints.zip','107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee','package-file-manifest.json')
    final_archive=checked_archive(FINAL/'final-results.zip','50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3','package-file-manifest.json')
    # The Kev result archive is retained inside the original Actions artifact.
    kev_result=KEV/'artifact/result.zip'
    kev_archive=checked_archive(kev_result,'1261b695cd6f1e1b429880cb0554fe3ff7f6c8231d3e70e6429bc8505a7bb566','file-manifest.json')
    qwen_summary=read(SOURCE/'docs/omni-products/v1/qwen25-research-summary.json')
    qwen_archive=checked_archive(QWEN/'original-provider-artifact.zip',qwen_summary['original_artifact']['sha256'])
    frozen=SOURCE/'docs/omni-products/v1'
    selected=read(frozen/'model-selection.json')
    metrics=read(frozen/'metrics.json')
    data=read(frozen/'data-definition.json')
    opening=read(frozen/'final-opening.json')
    values={part:read(FINAL/'verified/measurements'/part/'results.json')for part in ('av','sound','choice','integration')}
    for part in values:
        core_report.validate_part(part,FINAL/'verified/measurements'/part,opening,sha(frozen/'final-opening.json'))
    strict_original = {'passed':True}
    try:
        core_report.validate_complete(values,selected,metrics,data)
    except ValueError as error:
        strict_original={'passed':False,'error':str(error)}
    # Preserve every original file. Only a local comparison copy receives
    # recomputed aggregates after exact discrete equality and tiny float checks.
    replay=copy.deepcopy(values)
    comparison=measured.paired_AV_comparison(values['av']['models'])
    equal_or_small_float(values['av']['paired_family_comparison'],comparison,'/AVpaired')
    replay['av']['paired_family_comparison']=comparison
    for name,temperature,threshold in [('primary',selected['temperature'],selected['threshold']),('data_control_raw_only',1.0,None)]:
        actual=values['choice'][name]
        recomputed=measured.evaluate_c(actual['rows'],temperature,threshold)
        equal_or_small_float(actual,recomputed,'/C/'+name)
        replay['choice'][name]=recomputed
    core_report.validate_complete(replay,selected,metrics,data)
    verdict=core_report.verdict(replay,selected,metrics,data)
    assert verdict==read(FINAL/'verified/final-report.json')
    summary=read(SOURCE/'docs/omni-products/v1/results-summary.json')
    assert summary['capability_verdict']==verdict and verdict['global_targets_all_met'] is False and verdict['promote_C_service'] is False
    for key,source in [('AV','av'),('sound','sound'),('C','choice'),('integration','integration')]:
        # Summaries omit bulky raw rows and may add explicitly labelled prose.
        if key=='C':
            equal_or_small_float(summary[key]['primary']['types'],values[source]['primary']['types'],'/summary/C/types')
            assert summary[key]['primary']['policy']['accepted']==53 and summary[key]['primary']['policy']['errors']==19
    c=values['choice']['primary']
    assert c['types']['choice']['primary']['correct']==39 and c['types']['choice']['primary']['families']==60
    assert c['policy']['accepted']==53 and c['policy']['errors']==19
    # Prove missing data and false zero-risk summaries cannot pass this same
    # completeness checker, without changing any retained original measurement.
    rejections=[]
    for variant in ('missing_AV_family','changed_C_gold','false_C_zero_errors'):
        bad=copy.deepcopy(replay)
        if variant=='missing_AV_family':
            bad['av']['models'][selected['selected']['av']]['records'].pop()
        elif variant=='changed_C_gold':
            bad['choice']['primary']['rows'][0]['tasks']['choice']['gold']='unregistered'
        else:
            bad['choice']['primary']['policy']['errors']=0
        try:
            core_report.validate_complete(bad,selected,metrics,data)
        except ValueError as error:
            rejections.append({'case':variant,'rejected':True,'error':str(error)})
        else:
            raise AssertionError('invalid rows accepted: '+variant)
    kev_result_json=read(KEV/'verified/research-result.json')
    kev_job=read(KEV/'artifact/job.json')
    with zipfile.ZipFile(kev_result)as archive:
        research_measurements.check_kev(archive,kev_result_json,kev_job)
    assert kev_result_json['completed'] is True and kev_result_json['families_per_language']==60
    assert kev_result_json['language_views_share_same60_families'] is True and kev_result_json['own_training_used_Kev_outputs'] is False
    kev_views={}
    for language in ('english','chinese'):
        view=read(KEV/'verified'/('kev-final-'+language+'.json'))
        kev_views[language]={name:{'choice_correct':value['types']['choice']['primary']['correct'],'families':value['types']['choice']['primary']['families'],'accepted':value['policy']['accepted'],'errors':value['policy']['errors']}for name,value in view.items()}
    ledger=read(QWEN/'native-terminal-ledger.json')
    finance.validate_state(ledger,datetime.now(UTC))
    assert ledger['reserved_total_usd']=='14.00'
    groups={name:[row for row in ledger['attempts']if json.loads(row['purpose']).get('job')==name]for name in ('mature-kev','mature-qwen25','cpu-core-final-v1')}
    assert [row['status']for row in groups['mature-kev']]==['failed-stopped','completed']
    assert len(groups['mature-qwen25'])==1 and groups['mature-qwen25'][0]['status']=='failed-stopped'
    assert len(groups['cpu-core-final-v1'])==1 and groups['cpu-core-final-v1'][0]['status']=='completed'
    producer_sources=[]
    for label,folder in [('Kev',KEV/'artifact'),('Qwen',QWEN/'artifact')]:
        sources=read(folder/'sources.json')
        reserved=read(folder/'reservation.json')['attempt']
        receipt=read(folder/'research-receipt.json')
        research_execution.check_receipt_identity('kev' if label=='Kev'else'qwen25',reserved['attempt_id'],sources,reserved,receipt)
        for name,digest in sources['files'].items():
            raw=subprocess.check_output(['git','show',sources['revision']+':'+name],cwd=MODELS)
            assert hashlib.sha256(raw).hexdigest()==digest
        producer_sources.append({'label':label,'revision':sources['revision'],'actual_Git_source_files':len(sources['files']),'receipt_binds_original_attempt_source_input_opening':True})
    qwen=read(QWEN/'artifact/research-receipt.json')
    assert qwen['success'] is False and qwen['phase']=='waiting_for_task_id' and qwen['error_type']=='TimeoutError'
    assert not qwen.get('input_payload_uploaded') and not qwen.get('result_exported_before_stop')
    assert [a['name']for a in qwen['assets']]==['input.zip']
    assert qwen['compute_stopped'] is True and qwen['resources_verified'] is True
    assert qwen['remaining_owned_sandboxes']==[] and qwen['stop_errors']=={} and all(qwen['cleanup_checks'].values())
    assert qwen_summary['actual_execution']['accuracy'] is None and qwen_summary['actual_execution']['actual_model_dtype'] is None
    assert qwen_summary['budget']['automatic_retry_permitted'] is False and qwen_summary['budget']['attempt_reserved_usd']=='3.10'
    terminal_rejections=[]
    with patch.object(research_execution.research_recovery,'api',forbidden):
        for job in ('kev','qwen25'):
            try:
                research_execution.research_recovery.eligibility(ledger,job)
            except ValueError as error:
                terminal_rejections.append({'job':job,'rejected_without_provider_access':True,'error':str(error)})
            else:
                raise AssertionError('terminal research unexpectedly became eligible')
    assert attempts==[]

out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),
     'original_archives':{'core':core_archive,'final':final_archive,'Kev':kev_archive,'Qwen_original_Actions_artifact':qwen_archive},
     'four_final_origin_completion_and_opening_bindings_valid':True,'raw_strict_cross_host_result':strict_original,
     'pure_recomputed_aggregates':{'all_discrete_fields_exact':True,'float_tolerance_for_review_only':1e-11,'different_float_nodes':len(float_differences),'max_absolute_difference':max((v['absolute_difference']for v in float_differences),default=0),'original_files_and_scientific_checker_unchanged':True,'local_comparison_copy_passes_complete_original_checker':True},
     'original_final_report_reproduced':verdict,'C_observed':{'correct':39,'families':60,'accepted':53,'accepted_errors':19},
     'invalid_result_rejections':rejections,'Kev_actual_archive_complete_checker_pass':True,'Kev_views':kev_views,
     'Kev_language_views_share60_families_not120_independent':True,'Kev_saved260_raw_rows_cal80_policy60_final2x60_checked':True,
     'Qwen_failed_stopped_startup_only':{'phase':qwen['phase'],'error_type':qwen['error_type'],'input_uploaded':False,'result_archive_present':False,'no_accuracy_dtype_latency_or_memory_result_inferred':True,'source_Git_receipt_bindings':producer_sources},
     'terminal_ledger':{'preserved_source_snapshot_not_fresh_provider_query':True,'attempts':len(ledger['attempts']),'retained_bound_usd':ledger['reserved_total_usd'],'pure_validate_state_pass':True,'not_actual_invoice':True,'repeat_rejections':terminal_rejections},
     'forward_guard_selftested':True,'actual_review_model_forward_attempts':attempts,
     'limits':['Only saved rows/logits and original archive/receipt bytes were inspected. No neural final/teacher inference, train, model weight load, Modal/provider call, new permit, repeat, reservation or publication.','Read-only Git objects confirm original producer source versions, not an unobserved provider telemetry stream or new capability generalization.','Tiny cross-host float comparison tolerance applies only to independent numerical review copies; original exact checker/thresholds/files are preserved.','Public14 exact streaming download proof is separate. No raw packages, weights or third-party full texts are written to Git.']}
target=Path(__file__).with_name('scope450-original-results-inspection-result-v1.json')
with target.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'pass','final_payloads':final_archive['payloads_verified'],'Kev_payloads':kev_archive['payloads_verified'],'float_nodes_different':len(float_differences),'max_float_difference':out['pure_recomputed_aggregates']['max_absolute_difference'],'original_C_promote':verdict['promote_C_service'],'retained_bound_usd':ledger['reserved_total_usd'],'terminal_repeat_rejections':2,'model_forward_attempts':0},ensure_ascii=False))
