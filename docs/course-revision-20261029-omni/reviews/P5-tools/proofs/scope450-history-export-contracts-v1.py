"""Current historical-original, figure regeneration and true Git/Colab contracts."""
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts import export_course

ROOT=Path(__file__).resolve().parents[5]
BASE=Path('docs/course-revision-20261029-omni/reviews/P5-tools/proofs')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


fixture=module('p5_scope450_history_v2',ROOT/'tests/test_historical_rechecks_v2.py')
histories=[]
for label,variant,multi in [('complete_continuity',None,False),('honest_revise_original','honest_revise',False),('two_page_dependency',None,True)]:
    with tempfile.TemporaryDirectory(prefix='p5-history-positive-')as directory:
        root=Path(directory)
        built=fixture.fixture(root,variant,multi)
        before={n:(root/n).read_bytes()for n in ('prior/dispositions.json','prior/fixed.json')}
        fixture.check(root,built)
        assert all((root/n).read_bytes()==raw for n,raw in before.items())
        histories.append({'case':label,'accepted':True,'old_original_and_disposition_unchanged':True})
for variant in ['reader_absent','reader_stale','technical_original_shell','technical_original_created_after_current','technical_original_header_wrong_source','continuity_current_created_before_original','continuity_original_unbound_source','continuity_original_checks_empty','continuity_original_quote_absent','downgrade_necessity','historical_verification_replaced','duplicate_chain','nested_chain','stages_overlap','unavailable_listing_nonempty','prior_gate_failed','missing_page_evidence']:
    with tempfile.TemporaryDirectory(prefix='p5-history-negative-')as directory:
        root=Path(directory)
        try:
            fixture.check(root,fixture.fixture(root,variant))
        except (ValueError,KeyError)as error:
            histories.append({'case':variant,'accepted':False,'error':str(error)})
        else:
            raise AssertionError('invalid history accepted '+variant)

visuals=module('p5_scope450_visuals',ROOT/'scripts/build_visuals.py')
with tempfile.TemporaryDirectory(prefix='p5-preserve-normalization-')as directory:
    private=Path(directory)
    shutil.copytree(ROOT/'course/figures',private/'course/figures')
    metadata=Path('docs/lesson-products/results/publication-spec-v1.json')
    (private/metadata).parent.mkdir(parents=True)
    shutil.copyfile(ROOT/metadata,private/metadata)
    before={p.name:sha(p)for p in (private/'course/figures').glob('*.svg')}
    visuals.ROOT=private
    visuals.DIRECTORY=private/'course/figures'
    with contextlib.redirect_stdout(io.StringIO()):
        visuals.build()
    after={p.name:sha(p)for p in (private/'course/figures').glob('*.svg')}
    assert before==after and len(before)==215
    assert (private/'course/figures/index.json').read_bytes()==(ROOT/'course/figures/index.json').read_bytes()
    normalization=(private/'course/figures/normalization.svg').read_text()
    assert '縮放／偏移'in normalization and 'gamma；LN有beta'in normalization and '可學縮放'not in normalization
    figure={'SVG_count':215,'all_original_SVG_bytes_preserved':True,'index_exact_bytes_preserved':True,'normalization_sha256':before['normalization.svg'],'correct_normalization_labels_preserved':True,'private_rebuild_only':True}

colab=[]
with tempfile.TemporaryDirectory(prefix='p5-true-Git-Colab-')as directory:
    private=Path(directory)
    paths=['notebooks/1/a.ipynb','notebooks/1/b.ipynb']
    for name in paths:
        path=private/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[]}))
    def git(*args):
        return subprocess.check_output(['git','-C',str(private),*args],stderr=subprocess.STDOUT).decode().strip()
    git('init','-q')
    git('add','notebooks')
    git('-c','user.name=Private fixture','-c','user.email=fixture@example.invalid','commit','-qm','Actual two-Notebook fixture')
    revision=git('rev-parse','HEAD')
    index=[{'notebook':name}for name in paths]
    with patch.object(export_course,'ROOT',private):
        export_course.verify_colab_notebooks(index,revision)
        colab.append({'case':'actual_commit_two_current_Notebook_blobs','accepted':True})
        negatives=[('symbolic_main','main',index),('missing_commit','0'*40,index),('blob_not_commit',git('rev-parse',revision+':'+paths[0]),index),('missing_Notebook',revision,[*index,{'notebook':'notebooks/missing.ipynb'}])]
        for label,pin,items in negatives:
            try:
                export_course.verify_colab_notebooks(items,pin)
            except ValueError as error:
                colab.append({'case':label,'accepted':False,'error':str(error).splitlines()[0]})
            else:
                raise AssertionError('bad pin accepted '+label)
        (private/paths[0]).write_text('changed current Notebook')
        try:
            export_course.verify_colab_notebooks(index,revision)
        except ValueError as error:
            colab.append({'case':'current_Notebook_differs_from_real_blob','accepted':False,'error':str(error).splitlines()[0]})
        else:
            raise AssertionError('stale Notebook pin accepted')
    item={'notebook':paths[0]}
    book={'cells':[{'cell_type':'code','source':['print(1)\n'],'metadata':{}}]}
    preview=export_course.section_footer(item,book,'main',True,direct_colab=False)
    assert 'https://colab.research.google.com/'in preview and '/github/'not in preview
    assert 'id="colab-upload-instructions"'in preview and '在首頁按 **Upload notebook**'in preview and '若已在 Notebook 畫面'in preview and '**File → Upload notebook**'in preview
    assert 'href="'+paths[0]+'" download'in preview and '雲端環境未經本課實測'in preview
    formal=export_course.section_footer(item,book,revision,True,direct_colab=True)
    assert '/blob/'+revision+'/'+paths[0]in formal and 'colab-upload-instructions'not in formal

# Real current331 inputs must fail against the old integrated HEAD until an
# actual publication commit includes all current Notebook bytes. No ref change.
real_index=json.loads((ROOT/'course/lesson-index.json').read_bytes())
assert len(real_index)==331
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
actual_pin={'actual_local_HEAD':head,'current_Notebooks':331}
try:
    export_course.verify_colab_notebooks(real_index,head)
except ValueError as error:
    actual_pin.update(accepted=False,rejected_changed_or_missing_Notebooks=len(str(error).splitlines())-1,error_header=str(error).splitlines()[0])
else:
    actual_pin.update(accepted=True)

baseline=module('p5_scope450_baseline',ROOT/'docs/review-tools/baseline_replay.py')
supplement=ROOT/'docs/course-revision-20261029-omni/baseline-supplements-v2.json'
started=time.monotonic()
result=baseline.replay(ROOT,json.loads(supplement.read_bytes())['files'])
assert not result['failures']
raw=json.dumps(result,ensure_ascii=False,sort_keys=True).encode()
outside=Path('/workspace/work/omni-P5-full-tools-review-inspection-raw-v6/baseline450-replay-v1.json')
with outside.open('xb')as stream:
    stream.write(raw)
baseline_receipt={'status':result['status'],'records':len(result['records']),'failures':result['failures'],'supplement_sha256':sha(supplement),'elapsed_seconds':time.monotonic()-started,'full_result_sha256':hashlib.sha256(raw).hexdigest(),'interpretation':'Real preserved baseline replay, not current369 all gate; full repetitive result is outside Git and can be reproduced with exact supplement.'}

out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),'history_controls':histories,'history_scope':'Complete private synthetic three-stage/schema2/old-original fixtures, not real current collection/seal','normalization_and_figure_regeneration':figure,'Colab_true_Git_object_controls':colab,'preview_UI_semantics':{'same_Notebook_download':True,'homepage_Upload_notebook_and_Notebook_File_menu_distinguished':True,'no_preview_stale_GitHub_direct_link':True,'formal_link_contains_verified_fixture_commit':True,'actual_Google_login_upload_runtime_or_visual_readability_claimed':False},'actual_current331_publication_pin':actual_pin,'preserved_baseline_replay':baseline_receipt,'personally_verified_source_semantics_issue_IDs':['AUTHOR-HISTORY-CONTINUITY-LINEAGE','AUTHOR-VISUALS-NORM-REGEN','AUTHOR-INTEGRATION-PIN-FIXTURE','UI-COLAB-01','R1-COLAB-PIN-01'],'Windows_issue_not_closed_by_this_proof':True,'limits':['No new current369 gate, stage collection/seal or activation. Root still must assemble actual lineage/current source/Notebook/figure/estimates and remaining stages.','Actual HTML/browser layout and authenticated Colab runtime are not certified by footer/code/Git tests.','No author files modified, repository refs changed, actual NN/final/Modal/provider call or paid operation. Only private fixture Git commits.']}
path=Path(__file__).with_name('scope450-history-export-contracts-result-v1.json')
with path.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'history_controls':len(histories),'figures_preserved':215,'Colab_controls':len(colab),'baseline_status':result['status'],'baseline_records':len(result['records']),'source_semantics_IDs':out['personally_verified_source_semantics_issue_IDs']},ensure_ascii=False))
