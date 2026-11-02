"""T.6 input-provenance and inference-export probes; no publishing or long training."""
from contextlib import redirect_stdout
from datetime import datetime, UTC
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from PIL import Image
import torch

from scripts import infer_modal
from scripts.course_release import inference_payload
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.modal_data import modal_example

ART=Path(__file__).resolve().parent
ROOT=ART.parents[3]
WORK=Path('/workspace/work/reference/omni-technical-t-6-v1/local-cpu')
torch.set_num_threads(2)
result={'created_at':datetime.now(UTC).isoformat(),'environment':{'python':platform.python_version(),
        'torch':str(torch.__version__),'device':'cpu','threads':'2'},'provenance_probes':[]}
original=infer_modal.generate_modal
tok=ByteTokenizer()

def wrapped(model,prefix,image,wave,tokens):
    result['provenance_probes'].append({'prefix_ids':prefix.tolist(),'decoded_prefix':tok.decode(prefix.tolist()),
        'image_shape':None if image is None else list(image.shape),
        'waveform_shape':None if wave is None else list(wave.shape),'max_new_tokens':tokens})
    return original(model,prefix,image,wave,tokens)

infer_modal.generate_modal=wrapped
old_argv=sys.argv
try:
    for task,args in [('vision',['--color','blue','--shape','circle','--prompt','shape?']),
                      ('audio',['--frequency','880','--prompt','pitch?']),
                      ('joint',['--color','red','--shape','square','--frequency','220','--prompt','joint?'])]:
        sys.argv=['scripts/infer_modal.py',str(WORK/f'{task}.pt'),*args,'--tokens','16','--device','cpu']
        output=io.StringIO()
        with redirect_stdout(output): infer_modal.main()
        result['provenance_probes'][-1]['actual_report']=json.loads(output.getvalue())
finally:
    sys.argv=old_argv
    infer_modal.generate_modal=original

Image.new('L',(32,48),128).save(WORK/'gray.png')
_,_,pixels,_=modal_example({'image':'gray.png','question':'read digits','answer':'12'},WORK,'vision')
result['grayscale_32x48_loader']={'source_mode':'L','source_size':[32,48],
                                'actual_tensor_shape':list(pixels.shape),'dtype':str(pixels.dtype)}
saved=torch.load(WORK/'joint.pt',weights_only=True)
clean=inference_payload(saved,{'verification':'local-only probe, never published'})
torch.save(clean,WORK/'joint-inference-only.pt')
result['inference_export']={'before_keys':sorted(saved),'after_keys':sorted(clean),
                           'optimizer_absent':'optimizer' not in clean,
                           'rng_absent':not any('rng' in key for key in clean),
                           'trainable_list_absent':'trainable_parameters' not in clean}
cmd=[sys.executable,'scripts/train.py','--task','joint','--checkpoint',str(WORK/'joint-inference-only.pt'),
     '--resume','--freeze','partial','--train','--steps','500','--device','cpu']
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
c=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
log=ART/'inference-only-resume-rejection.log'
log.write_text(c.stdout+c.stderr)
result['inference_only_resume']={'command':shlex.join(cmd),'returncode':c.returncode,
                               'log':str(log.relative_to(ROOT)),
                               'sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                               'rejected':c.returncode!=0}
assert c.returncode!=0
result['local_model_input_fingerprints']={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                                                'bytes':p.stat().st_size,'path':str(p)}
                                        for p in sorted(WORK.glob('*.pt'))}
(ART/'supplemental.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'provenance':[p['decoded_prefix'] for p in result['provenance_probes']],
                  'image_loader':result['grayscale_32x48_loader'],'inference_export':result['inference_export'],
                  'resume_returncode':c.returncode},ensure_ascii=False))
