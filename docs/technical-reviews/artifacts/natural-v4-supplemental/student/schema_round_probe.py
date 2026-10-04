"""New bounded offline checks needed to complete explicit software evidence."""
import base64
import io
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts import fetch_natural_release as fetch
from tiny_perceptron import natural_ui as ui

manifest=fetch.read_json(ROOT/'docs/natural-assistant/v4/public-release.json')
fixtures=ROOT/'docs/natural-assistant/evidence/v4-research/student-selected-public-ui/fetched-public-files'
result={'environment':{'python':sys.version,'torch':torch.__version__,'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__,'numpy':np.__version__,'device':'cpu'},'scope':'Offline small-file staging/retry/missing-file and four audio-format checks only; no installation, remote download or inference'}
with tempfile.TemporaryDirectory(prefix='schema-round-',dir=OUT) as temporary:
    temporary=Path(temporary);target=temporary/'release';calls=[]
    def fixture(repo,path,*,revision,token):
        item=next(x for x in manifest['files'] if x['path']==path)
        calls.append(path)
        return fixtures/(item['output']+'.raw')
    def interrupted(repo,path,*,revision,token):
        if path.endswith('release-provenance.json'):raise OSError('offline interruption fixture after first file')
        return fixture(repo,path,revision=revision,token=token)
    try:fetch.fetch_release(manifest,target,downloader=interrupted)
    except OSError as error:interruption=str(error)
    else:raise AssertionError('Injected interruption did not propagate')
    assert not target.exists()
    assert not list(temporary.glob('.natural-release-*'))
    fetched=fetch.fetch_release(manifest,target,downloader=fixture)
    assert fetch.verify_release(manifest,fetched)==target
    result['interrupted_stage_and_retry']={'observed_error':interruption,'final_target_absent_after_failure':True,'temporary_stage_removed':True,'same_output_retry_verified':True,'download_source':'existing original small public file fixtures; no network'}
    (target/'README.md').unlink()
    try:fetch.verify_release(manifest,target)
    except ValueError as error:missing=str(error)
    else:raise AssertionError('Missing file accepted')
    assert target.exists() and (target/'release-provenance.json').exists()
    result['missing_file']={'rejected':True,'observed_error':missing,'directory_remaining_files_preserved':True}

audio=[]
for format,suffix,subtype in [('WAV','.wav','PCM_16'),('FLAC','.flac','PCM_16'),('MP3','.mp3',None),('OGG','.ogg','VORBIS')]:
    stream=io.BytesIO()
    kwargs={'format':format}
    if subtype:kwargs['subtype']=subtype
    sf.write(stream,np.zeros(1600,dtype=np.float32),16000,**kwargs)
    payload=stream.getvalue()
    accepted,extension,media_type=ui.upload_contents({'kind':'audio','filename':'short'+suffix,'base64':base64.b64encode(payload).decode()})
    assert accepted==payload and extension==suffix
    info=sf.info(io.BytesIO(payload))
    audio.append({'format':format,'suffix':suffix,'bytes':len(payload),'decoded_frames':info.frames,'sample_rate':info.samplerate,'channels':info.channels,'actual_decoded_seconds':info.frames/info.samplerate,'accepted':True,'media_type':media_type})
result['audio_formats']=audio
result['status']='passed'
print(json.dumps(result,ensure_ascii=False,indent=2))
