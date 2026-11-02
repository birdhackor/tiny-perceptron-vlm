"""One same-owner anonymous GET, ZIP-member hashes, retained evidence identity.

No Authorization header, no ZIP/weights persisted, no signed redirect query
logged, no model inference/training/final/Modal operation.
"""
import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path('/tmp/tpb').resolve()
BASE = Path('docs/technical-reviews/artifacts/c-v3-v21-numbered-r4')
OUTPUT = BASE/'public-transport-current-v2.json'
COMMAND = 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-numbered-r4/verify-public-current-v2.py'
ORIGINAL = Path('docs/course-revision-20261101-c-v3/technical-v21/numbered_r4/original-v1.json')
CURRENT1 = ORIGINAL.with_name('current-v1.json')
ORIGINAL_SHA = '3ee1c43932862165c1b2f58a48d10e22eeaa5631628eab379f53e130aa638092'
CURRENT1_SHA = '5e634f883ea30e90f5698bde14c44a4fa11d47b8224806dccb131c6fa7d43237'
EXPECTED_ZIP_SHA = '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
EXPECTED_BYTES = 6168350
EXPECTED_URL = 'https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/c-v3-20261101/tiny-perceptron-c-v3-verified.zip'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024*1024):
            sha.update(chunk)
    return sha.hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


assert not (ROOT/OUTPUT).exists(), 'A saved one-GET receipt must not be replaced or retried'
assert sha_file(ROOT/ORIGINAL) == ORIGINAL_SHA
assert sha_file(ROOT/CURRENT1) == CURRENT1_SHA
original, current1 = read(ORIGINAL), read(CURRENT1)
relocation_path = Path('docs/course-revision-20261101-c-v3/tmpfs-source-identity-receipt-v1.json')
relocation = read(relocation_path)
assert Path(relocation['current_root']).resolve() == ROOT
assert relocation['old_root'] == '/workspace/work/omni-book-integrated-v1'
assert relocation['old_source_and_evidence_preserved_in_place'] is True
release = read('docs/decision-product/v3/public-release.json')
assert (release['url'],release['bytes'],release['sha256']) == (EXPECTED_URL,EXPECTED_BYTES,EXPECTED_ZIP_SHA)
root_event_path = Path('docs/course-revision-20261101-c-v3/evidence/public-release-anonymous-verified-v1.json')
assert sha_file(ROOT/root_event_path) == 'fab5cdf38e320f5f80e815c84b1f39067d7a4c96b7c864d36ecc84e697986868'
root_event = read(root_event_path)
assert root_event['source_commit'] == '70e1d9449e779c43cabfb5250f169106e573c74f'
sources = []
for source in current1['sources']:
    if source['kind'] == 'repository_code':
        observed = sha_file(ROOT/source['path'])
        assert observed == source['sha256'], source['path']
        sources.append({'path':source['path'],'sha256':observed})
artifacts = []
for artifact in current1['artifacts']:
    observed = sha_file(ROOT/artifact['path'])
    assert observed == artifact['sha256'], artifact['path']
    artifacts.append({'path':artifact['path'],'sha256':observed})
receipts = []
for suffix, report_path, report_sha in [('original-v1',ORIGINAL,ORIGINAL_SHA),('current-v1',CURRENT1,CURRENT1_SHA)]:
    path=Path('docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/numbered-r4-'+suffix+'.json')
    record=read(path)
    assert record['report']=={'path':str(report_path),'sha256':report_sha}
    assert record['reviewer_task']==original['reviewer_task']
    receipts.append({'path':str(path),'sha256':sha_file(ROOT/path),'received_at':record['received_at']})
live=(ROOT/'course/README.md').read_bytes()
start=live.index('## R.4 '.encode());end=live.find(b'\n## ',start+4);end=len(live) if end<0 else end+1
assert sha_bytes(live[start:end])==original['source_sha256']
section=Path('docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.4.md')
assert sha_file(ROOT/section)==original['source_sha256']
old_execution=read(BASE/'execution.json')
assert len(old_execution['fresh_cpu_examples'])==3
assert old_execution['result']=='All bounded assertions passed'
old_zip=next(row for row in old_execution['archives'] if row['sha256']==EXPECTED_ZIP_SHA)
local_zip=Path(old_zip['path'])
local_identity={'path':str(local_zip),'available':local_zip.is_file(),'retained_expected_sha256':old_zip['sha256'],'retained_expected_bytes':old_zip['bytes']}
if local_zip.is_file():
    local_identity.update(observed_bytes=local_zip.stat().st_size,observed_sha256=sha_file(local_zip))
    assert local_identity['observed_bytes']==EXPECTED_BYTES and local_identity['observed_sha256']==EXPECTED_ZIP_SHA

redirects=[]


def assert_no_authorization(request):
    headers={key.lower():value for key,value in request.header_items()}
    assert 'authorization' not in headers and 'proxy-authorization' not in headers


class AnonymousRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        assert_no_authorization(request)
        redirected=super().redirect_request(request,fp,code,message,headers,new_url)
        if redirected is not None:
            assert_no_authorization(redirected)
        # Never persist new_url, Location, or any query string.
        redirects.append({'status':code,'from_host':urlsplit(request.full_url).hostname,'to_host':urlsplit(new_url).hostname,'authorization_header_sent':False})
        return redirected


result={
    'schema_version':1,'review_stage':'technical','reviewer_task':original['reviewer_task'],'lesson_id':'R.4',
    'callback_kind':'same-owner public transport delta, not a new fresh original',
    'command':COMMAND,
    'environment':{'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'device':'CPU transport/hash only; no NN','platform':platform.platform()},
    'root':str(ROOT),
    'relocation_identity_receipt':{'path':str(relocation_path),'sha256':sha_file(ROOT/relocation_path)},
    'source_sha256':original['source_sha256'],'figure_sha256':{},
    'original_report':{'path':str(ORIGINAL),'sha256':ORIGINAL_SHA},
    'prior_current_report':{'path':str(CURRENT1),'sha256':CURRENT1_SHA},
    'prior_intake_receipts':receipts,
    'repository_sources_preserved':sources,'prior_artifacts_preserved':artifacts,
    'retained_local_zip':local_identity,
    'coordinator_trigger':{'path':str(root_event_path),'sha256':sha_file(ROOT/root_event_path),'workflow_run_id':root_event['workflow_run_id'],'source_commit':root_event['source_commit'],'role':'Trigger and publication chronology only; not a substitute for this reviewer GET.'},
    'request':{'url':EXPECTED_URL,'method':'GET','logical_get_attempts':1,'authorization_header_sent':False,'credential_parameters_supplied':False,'zip_or_weights_persisted':False,'signed_redirect_query_logged':False},
    'expected_bytes':EXPECTED_BYTES,'expected_sha256':EXPECTED_ZIP_SHA,
    'started_at':datetime.now(UTC).isoformat(),
    'transport_verified':False,'archive_members_verified':False,
    'scope':'One anonymous GET and in-memory ZIP/member verification; fixed local qualification and three prior CPU examples retained by identity. No NN/training/final/Modal, no network retry, no product/source/release mutation.'
}
payload=io.BytesIO()
try:
    opener=build_opener(AnonymousRedirects())
    request=Request(EXPECTED_URL,headers={'User-Agent':'R4-independent-public-delta-review','Accept':'application/octet-stream'},method='GET')
    assert_no_authorization(request)
    with opener.open(request,timeout=45) as response:
        observed_sha,count=hashlib.sha256(),0
        while chunk:=response.read(1024*1024):
            count+=len(chunk)
            observed_sha.update(chunk)
            payload.write(chunk)
            if count>EXPECTED_BYTES:
                raise ValueError('Response exceeded the immutable descriptor size')
        result.update(http_status=response.status,observed_bytes=count,observed_sha256=observed_sha.hexdigest())
        result['transport_verified']=response.status==200 and count==EXPECTED_BYTES and observed_sha.hexdigest()==EXPECTED_ZIP_SHA
    if result['transport_verified']:
        with zipfile.ZipFile(payload) as archive:
            prefix='tiny-perceptron-c-v3/'
            names=archive.namelist()
            assert len(names)==len(set(names)), 'Duplicate archive names'
            manifest_raw=archive.read(prefix+'MANIFEST.json')
            manifest=json.loads(manifest_raw)
            expected_names={prefix+'MANIFEST.json'}|{prefix+name for name in manifest['files']}
            assert {info.filename for info in archive.infolist() if not info.is_dir()}==expected_names
            members=[]
            for name,expected in manifest['files'].items():
                raw=archive.read(prefix+name)
                observed=sha_bytes(raw)
                assert len(raw)==expected['bytes'] and observed==expected['sha256'], name
                members.append({'member':prefix+name,'bytes':len(raw),'sha256':observed})
            for name,expected in release['bundle'].items():
                assert sha_bytes(archive.read(prefix+'bundle/'+name))==expected
            metadata=read('docs/decision-product/v3/metadata/acquisition.json')
            for row in metadata['files']:
                assert archive.read(row['archive_member'])==(ROOT/row['path']).read_bytes()
            for name,expected in release['implementation'].items():
                assert sha_bytes(archive.read(prefix+name))==expected==sha_file(ROOT/name)
            result.update(archive_members_verified=True,manifest_sha256=sha_bytes(manifest_raw),manifest_members=members,manifest_member_count=len(members),inference_files_verified=len(release['bundle']),metadata_files_byte_identical=len(metadata['files']),implementation_files_byte_identical=len(release['implementation']))
except HTTPError as error:
    result.update(http_status=error.code,error_type=type(error).__name__,error='HTTP '+str(error.code)+'; no response/redirect query persisted')
except (URLError,OSError,ValueError,AssertionError,zipfile.BadZipFile) as error:
    result.update(error_type=type(error).__name__,error=str(error) if not isinstance(error,URLError) else 'URL transport error; sensitive redirect query not logged')
finally:
    payload.close()
result['redirects']=redirects
result['finished_at']=datetime.now(UTC).isoformat()
result['verified']=result['transport_verified'] and result['archive_members_verified']
result['result']='Anonymous fixed public transport and all original ZIP members verified' if result['verified'] else 'Public transport or archive member verification remains unresolved'
assert sha_file(ROOT/ORIGINAL)==ORIGINAL_SHA and sha_file(ROOT/CURRENT1)==CURRENT1_SHA
(ROOT/OUTPUT).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'receipt':str(OUTPUT),'sha256':sha_file(ROOT/OUTPUT),'started_at':result['started_at'],'finished_at':result['finished_at'],'http_status':result.get('http_status'),'bytes':result.get('observed_bytes'),'asset_sha256':result.get('observed_sha256'),'verified':result['verified'],'manifest_member_count':result.get('manifest_member_count'),'authorization_header_sent':False},ensure_ascii=False))
