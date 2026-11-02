"""Read-only anonymous GET/hash of14 already published fixed assets; no model load."""
import concurrent.futures
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path('/workspace/work/omni-P5-full-tools-review-source-v6/docs/course-revision-20261029-omni/evidence/P5-public-assets-live-author-v1.json')
manifest = json.loads(SOURCE.read_bytes())
releases = manifest['releases']
assert {row['id'] for row in releases} == {428573191,428660172,428701352}
assets = [{**asset, 'release_id': release['id'], 'release_tag': release['tag_name']} for release in releases for asset in release['assets']]
assert len(assets) == 14 and sum(row['bytes'] for row in assets) < 150*1024**2


def get_asset(row):
    url = row['browser_download_url']
    assert url == 'https://github.com/birdhackor/tiny-perceptron-vlm/releases/download/'+row['release_tag']+'/'+row['name']
    assert 0 < row['bytes'] <= 128*1024**2 and row['digest'].startswith('sha256:')
    request = urllib.request.Request(url, headers={'User-Agent':'P5-independent-engineering-review','Accept':'application/octet-stream'})
    assert not request.has_header('Authorization')
    proof = {'asset_id':row['id'],'release_id':row['release_id'],'name':row['name'],'url':url,'expected_bytes':row['bytes'],'expected_sha256':row['digest'][7:],'request_authorization_header_present':False}
    try:
        count, digest = 0, hashlib.sha256()
        with urllib.request.urlopen(request, timeout=45) as response:
            proof.update(status=response.status, final_host=urllib.parse.urlparse(response.url).hostname)
            while block := response.read(1024**2):
                count += len(block)
                if count > row['bytes']:
                    raise ValueError('published asset exceeds exact expected bound')
                digest.update(block)
        proof.update(received_bytes=count,actual_sha256=digest.hexdigest(),exact_bytes_and_sha_match=count==row['bytes'] and digest.hexdigest()==row['digest'][7:])
    except Exception as error:
        proof.update(error_type=type(error).__name__, status=getattr(error,'code',None), exact_bytes_and_sha_match=False)
    return proof


started = datetime.now(UTC).isoformat()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(get_asset, assets))
out = {'reviewer_task':'/root/omni_p5_full_tools_review','started_at':started,'completed_at':datetime.now(UTC).isoformat(),'source_receipt_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'asset_count':len(results),'all14_actual_anonymous_GET_bytes_SHA_verified':all(row['exact_bytes_and_sha_match'] for row in results),'results':results,'weights_or_raw_archives_saved_to_Git':False,'model_load_forward_or_final':False,'provider_mutation':False,'limits':'Fresh read-only anonymous GET streaming hashes. Bytes discarded after hashing; no model deserialization/inference, training, publication, retry of terminal jobs, permit, reservation or paid compute.'}
target = Path(__file__).with_name('scope450-anonymous-public-downloads-result-v1.json')
with target.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all14_actual_anonymous_GET_bytes_SHA_verified':out['all14_actual_anonymous_GET_bytes_SHA_verified'],'results':[{'name':row['name'],'status':row['status'],'match':row['exact_bytes_and_sha_match']}for row in results]},ensure_ascii=False))
