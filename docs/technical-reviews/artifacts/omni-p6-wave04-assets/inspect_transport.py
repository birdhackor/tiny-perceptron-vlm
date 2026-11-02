"""Exercise only file transport contracts with tiny artificial bytes, no models."""

import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision.public_download import install
from scripts.selftrained.train_local_stage import local_source

OUT = Path(__file__).parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    prefix = 'core/fixture/'
    content = {
        prefix+'training/weights.safetensors': b'artificial tensor file bytes; never parsed',
        prefix+'training/training-result.json': b'{"fixture":true}\n',
        prefix+'job-result.json': b'{"fixture":true}\n',
        prefix+'training/latest.pt': b'not a pickle; no deserialization permitted',
    }
    manifest = {name: {'bytes':len(raw),'sha256':sha(raw)} for name,raw in content.items()}
    expected = {name:manifest[name] for name in content if not name.endswith('.pt')}
    with tempfile.TemporaryDirectory(prefix='assets-review-transport-') as temporary:
        root = Path(temporary)
        archive = root/'fixture.zip'
        with zipfile.ZipFile(archive,'w') as package:
            for name,raw in content.items():
                package.writestr(name,raw)
            package.writestr('package-file-manifest.json',json.dumps(manifest))
        row = {'job':'fixture','bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes())}
        inference = install(archive,row,root/'inference',False,expected_files=expected)
        full = install(archive,row,root/'full',True,expected_files=expected)
        assert set(inference)==set(expected) and set(full)==set(content)
        assert not (root/'inference'/f'{prefix}training/latest.pt').exists()
        assert (root/'full'/f'{prefix}training/latest.pt').read_bytes()==content[f'{prefix}training/latest.pt']
        bad_expected = {name:dict(value) for name,value in expected.items()}
        bad_expected[f'{prefix}training/weights.safetensors']['sha256']='0'*64
        try:
            install(archive,row,root/'mismatch',False,expected_files=bad_expected)
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError('Changed selected weights must be refused')
        assert not (root/'mismatch').exists()
        rejections=[]
        for name,resume in [('model.safetensors',False),('latest.pt',False),('best.pt',True)]:
            try:
                local_source(root/name,'0'*64,resume)
            except ValueError as error:
                rejections.append({'name':name,'resume':resume,'error':str(error)})
            else:
                raise AssertionError('Wrong checkpoint type must be refused')
        assert 'torch' not in sys.modules
        result={
            'created_at':datetime.now(UTC).isoformat(),
            'environment':{'python':sys.version,'platform':platform.platform(),'device':'no tensor allocation; standard library only'},
            'fixture_archive_bytes':row['bytes'],
            'default_installed_files':sorted(inference),'full_installed_files':sorted(full),
            'selected_weight_mismatch_rejected':rejection,'mismatch_wrote_files':False,
            'local_stage_wrong_source_rejections':rejections,
            'torch_imported':False,
            'scope':'Tiny reviewer-created ZIP, not a published neural checkpoint. Confirms file-layer installation split, binding refusal and wrapper source-name guard only; does not prove model tensor validity, full training resume, cross-platform parity or model quality.'
        }
    (OUT/'transport-inspection-v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
