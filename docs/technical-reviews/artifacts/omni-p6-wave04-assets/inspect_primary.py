"""Recompute small public-metadata bindings and preserve short original excerpts."""

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
CACHE = Path('/workspace/work/reference/omni-p6-wave04-assets')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    primary = json.loads((OUT/'primary-receipts-v1.json').read_bytes())['sources']
    primary += json.loads((OUT/'primary-extra-receipts-v1.json').read_bytes())['sources']
    primary += json.loads((OUT/'legacy-primary-receipts-v1.json').read_bytes())['sources']
    receipts = {row['name']:row for row in primary}
    for name,row in receipts.items():
        assert sha(CACHE/name)==row['sha256']
    selected = json.loads((ROOT/'docs/omni-products/v1/public-release.json').read_bytes())
    release = json.loads((CACHE/'github-release-api.json').read_bytes())
    assert release['draft'] is False and release['tag_name']==selected['release_tag']
    release_bindings={}
    for kind,row in selected['components'].items():
        remote=next(item for item in release['assets'] if item['id']==row['asset_id'])
        assert remote['name']==row['name'] and remote['size']==row['bytes']
        assert remote['digest']=='sha256:'+row['sha256']
        assert remote['state']=='uploaded'
        release_bindings[kind]={key:remote[key] for key in ['id','name','size','digest','state','browser_download_url']}

    v2_tree=json.loads((CACHE/'hf-v2-tree.json').read_bytes())
    v2_files={row['path']:row for row in v2_tree if row['type']=='file'}
    names=['moe-pretrain','moe-sft','moe-joint','dense-joint']
    expected_files={'model.safetensors','model-config.json','tokenizer.json','inference-manifest.json'}
    assert set(v2_files)=={f'selftrained/v2/{component}/{name}' for component in names for name in expected_files}
    release_definition=json.loads((ROOT/'docs/selftrained/v2-jobs/release-final-four-exports.json').read_bytes())['release']
    licensed={item['name']:item for item in release_definition['exports']}
    v2_bindings={}
    for component in names:
        manifest=json.loads((CACHE/f'hf-v2-{component}-inference-manifest.json').read_bytes())
        assert set(manifest['files'])==expected_files-{'inference-manifest.json'}
        assert not {'optimizer','rng','sampler'} & set(manifest)
        config=json.loads((CACHE/f'hf-v2-{component}-model-config.json').read_bytes())
        assert config['architecture']==('dense' if component=='dense-joint' else 'moe')
        bindings={}
        for name in sorted(expected_files):
            remote=v2_files[f'selftrained/v2/{component}/{name}']
            authored=next(item for item in licensed[component]['files'] if item['path']==name)
            assert authored['license']=='MIT' and remote['size']==authored['bytes']
            if name=='model.safetensors':
                assert remote['lfs']['oid']==authored['sha256']==manifest['files'][name]
                digest=remote['lfs']['oid']
            else:
                cached=CACHE/f'hf-v2-{component}-{name}'
                assert sha(cached)==authored['sha256']
                if name!='inference-manifest.json':
                    assert sha(cached)==manifest['files'][name]
                digest=sha(cached)
            bindings[name]={'bytes':remote['size'],'sha256':digest,'weight_digest_scope':'provider LFS OID, no tensor GET' if name=='model.safetensors' else 'actual metadata GET SHA-256'}
        v2_bindings[component]={'stage':manifest['stage'],'files':bindings,'export_manifest_state_keys':sorted(set(manifest)&{'optimizer','rng','sampler'}),'architecture':config['architecture']}
    current_info=json.loads((CACHE/'hf-current-model-info.json').read_bytes())
    assert current_info['cardData']['license']=='mit'
    current_card=(CACHE/'hf-current-root-readme.md').read_text()
    licence_line=next(line for line in current_card.splitlines() if '共16個檔案' in line and 'MIT' in line)
    assert '979cdfacc588ad0536f1c64fff96f264571cf054' in licence_line

    natural=json.loads((ROOT/'docs/natural-assistant/v4/public-release.json').read_bytes())
    natural_tree={row['path']:row for row in json.loads((CACHE/'hf-natural-v4-tree.json').read_bytes())}
    assert set(natural_tree)=={row['path'] for row in natural['files']}
    for item in natural['files']:
        cache=CACHE/('hf-natural-v4-'+item['output'])
        assert natural_tree[item['path']]['size']==item['bytes'] and sha(cache)==item['sha256']
    natural_provenance=json.loads((CACHE/'hf-natural-v4-release-provenance.json').read_bytes())
    assert natural_provenance['selected_variant']=='base'
    assert natural_provenance['base_model']==natural['base_model']
    assert natural_provenance['asr_model']==natural['asr_model']

    legacy=json.loads((CACHE/'hf-legacy-tree.json').read_bytes())
    legacy_files={row['path']:row for row in legacy if row['type']=='file'}
    public=json.loads((ROOT/'docs/course-experiments/public-models.json').read_bytes())
    legacy_declared=[item for model in public['models'] for item in model['files']]
    matching_size=0
    matching_lfs_digest=0
    for model in public['models']:
        pinned_tree={row['path']:row for row in json.loads((CACHE/f'hf-course-{model["id"]}-tree.json').read_bytes()) if row['type']=='file'}
        for item in model['files']:
            row=pinned_tree[item['path']]
            assert row['size']==item['bytes']
            matching_size+=1
            if 'lfs' in row:
                assert row['lfs']['oid']==item['sha256']
                matching_lfs_digest+=1
    capstone=json.loads((ROOT/'docs/course-experiments/capstone-public.json').read_bytes())
    capstone_files={item['path']:item for model in capstone['models'] for item in model['files']}
    for name,item in capstone_files.items():
        assert legacy_files[name]['size']==item['bytes']
        if 'lfs' in legacy_files[name]:
            assert legacy_files[name]['lfs']['oid']==item['sha256']

    from tiny_perceptron.data import toy_documents,toy_conversations,split_documents
    toy={'documents':len(toy_documents()),'conversations':len(toy_conversations()),'same_repeated_generation':toy_documents()==toy_documents() and toy_conversations()==toy_conversations(),'same_seed_split':split_documents(toy_documents(),seed=42)==split_documents(toy_documents(),seed=42)}
    result={
        'created_at':datetime.now(UTC).isoformat(),
        'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU, metadata/strings only; no model construction or forward'},
        'public_release':{'id':release['id'],'draft':release['draft'],'prerelease':release['prerelease'],'published_at':release['published_at'],'bindings':release_bindings},
        'v2_files':v2_bindings,'v2_file_count':len(v2_files),'v2_total_bytes':sum(row['size'] for row in v2_files.values()),
        'licensor_current_card':{'url':receipts['hf-current-root-readme.md']['url'],'model_api_commit':current_info['sha'],'response_commit':receipts['hf-current-root-readme.md']['response_headers'].get('X-Repo-Commit'),'full_raw_sha256':receipts['hf-current-root-readme.md']['sha256'],'excerpt':licence_line,'scope':'Licensor\'s public licence and designated format statement only; model-quality/test summaries are not evidence for this asset review.'},
        'natural_public':{'tree_file_count':len(natural_tree),'tree_bytes':sum(row['size'] for row in natural_tree.values()),'base_model':natural['base_model'],'asr_model':natural['asr_model'],'selected_variant':natural_provenance['selected_variant'],'no_copied_base_or_asr_weight_files':True,'metadata_file_sha_matches':True},
        'legacy_metadata':{'course_models':len(public['models']),'declared_file_entries':len(legacy_declared),'provider_size_matches':matching_size,'provider_lfs_oid_matches':matching_lfs_digest,'capstone_models':len(capstone['models']),'capstone_unique_files':len(capstone_files),'scope':'Each of 30 model namespaces checked at its own manifest pin; capstone checked at 33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed. Metadata/retention only; no fresh inference or download of all historical non-LFS payloads.'},
        'synthetic_generators':toy,
        'original_excerpts':[
            {'name':'lfs-spec-v3.6.1.md','locator':'The Pointer / required keys; Intercepting Git','excerpt':'The core Git LFS idea is that instead of writing large blobs to a Git repository, only a pointer file is written. / Currently, only `sha256` is supported. / `size` is in bytes. / Note that the `clean` filter does not push the file to the server. Use the `git push` command to do that (lfs files are pushed before commits in a pre-push hook).'},
            {'name':'lfs-smudge-v3.6.1.adoc','locator':'OPTIONS','excerpt':'`--skip`:: Skip automatic downloading of objects on clone or pull. `GIT_LFS_SKIP_SMUDGE`:: Disables the smudging process.'},
            {'name':'github-lfs-billing.html','locator':'How use of Git LFS is measured; Usage in forks','excerpt':'If you make a 1 byte change and push the file again, you\'ll use another 500 MB of storage / If you download a 500 MB file that\'s tracked with Git LFS, you\'ll use 500 MB of the repository owner\'s bandwidth. / Bandwidth and storage usage always count against the repository owner\'s account.'},
            {'name':'hf-storage-limits.html','locator':'Storage limits / Storage plans','excerpt':'We do have mitigations in place to prevent abuse of free public storage / Beyond the first few gigabytes, please use this resource responsibly by uploading content that offers genuine value to other users. / We also bill for storage space for private repositories, above a free tier.'},
            {'name':'pytorch-save-load.html','locator':'Saving & Loading a General Checkpoint for Inference and/or Resuming Training','excerpt':'When saving a general checkpoint, to be used for either inference or resuming training, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains.'},
            {'name':'pytorch-randomness.html','locator':'Reproducibility / opening; Controlling sources of randomness','excerpt':'Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.'},
            {'name':'peft-checkpoint.html','locator':'PEFT files / adapter_model and adapter_config','excerpt':'to load a PEFT model, you need to have the original model available as well. / The state_dict only contains the parameters of the adapter module, not the base model. / information about the base model like "base_model_name_or_path" / the revision of the model (if any)'}
        ],
        'scope':'Actual bounded unauthenticated public metadata, fixed local declarations, original official docs and tiny deterministic data-generator calls. No weight payload read, neural forward/training, full model download or publication.'
    }
    (OUT/'primary-inspection-v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'v2_file_count':result['v2_file_count'],'v2_total_bytes':result['v2_total_bytes'],'natural_public':result['natural_public'],'legacy_metadata':result['legacy_metadata'],'licence_commit':current_info['sha'],'synthetic_generators':toy},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
