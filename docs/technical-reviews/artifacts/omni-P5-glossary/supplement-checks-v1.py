"""Additional scalar/array checks and read-only publication metadata inspection."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from urllib.request import Request, urlopen

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
PROOF = Path(__file__).resolve().parent
OUTSIDE = Path('/workspace/work/reference/omni-P5-glossary-whole')


def main():
    results = {}
    scalar = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    loss = (2 * scalar + 1).square()
    loss.backward()
    assert scalar.grad.item() == 28
    results['backward_chain_rule'] = {'input': 3, 'formula': '(2*x+1)^2', 'loss': loss.item(),
                                      'gradient': scalar.grad.item(), 'hand_calculation': '2*(2*3+1)*2=28'}
    a = torch.tensor([1.2, -.4], dtype=torch.float64)
    first, second = torch.tensor([1., 0.], dtype=torch.float64), torch.tensor([.25, -.5], dtype=torch.float64)
    results['rvq_arithmetic'] = {'residual_after_first': (a-first).tolist(), 'sum_reconstruction': (first+second).tolist(),
                                  'remaining_error': (a-first-second).tolist()}
    from tiny_perceptron.multimodal import patchify, mel_filter_bank, tone, log_mel
    image = torch.arange(3*8*8).reshape(1,3,8,8)
    patches = patchify(image, 4)
    assert patches.shape == (1, 4, 48)
    mel = mel_filter_bank()
    assert mel.shape == (16, 201) and (mel >= 0).all() and (mel.max(0).values > 0).any()
    wave = tone(440, .1, 16000)
    assert len(wave) == 1600
    spectrum = log_mel(wave)
    assert spectrum.shape[0] == 16
    results['patch_and_spectrogram'] = {'input': list(image.shape), 'patches': list(patches.shape),
                                       'mel_bank': list(mel.shape), 'tone_frequency_hz': 440,
                                       'sample_rate': 16000, 'duration_seconds': .1, 'samples': len(wave),
                                       'log_mel_axes': list(spectrum.shape), 'scope': 'deterministic raw signal/array operations; no learned model'}
    from tiny_perceptron.omni_products.av import local_audio
    zero = torch.zeros(700*16)
    features, starts, ends, anchors = local_audio(zero)
    assert starts.tolist() == [0., 200., 400., 600.] and ends.tolist() == [200.,400.,600.,700.]
    changed = zero.clone(); changed[3200:] = .5
    f2 = local_audio(changed)[0]
    assert torch.equal(features[0], f2[0])
    results['local_support'] = {'feature_shape': list(features.shape), 'starts_ms': starts.tolist(),
                                'ends_ms': ends.tolist(), 'anchors_ms': anchors.tolist(),
                                'outside_first_block_perturbation_preserves_first_feature': True}
    release = json.loads((ROOT / 'tiny_perceptron/lesson_products/release.json').read_text())
    publication = []
    for product, spec in release['products'].items():
        url = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{spec['prefix']}/artifact-manifest.json"
        with urlopen(Request(url, headers={'User-Agent':'Independent educational review'}), timeout=25) as response:
            raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == spec['manifest_sha256']
        manifest = json.loads(raw)
        assert manifest['product'] == product
        publication.append({'product': product, 'url': url, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'accessed_at': datetime.now(timezone.utc).isoformat(), 'payload_files': list(manifest['files']),
                            'weights_downloaded': False, 'inference_run': False})
    a_url = 'https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint/inference-manifest.json'
    with urlopen(Request(a_url, headers={'User-Agent':'Independent educational review'}), timeout=25) as response:
        raw = response.read()
    (OUTSIDE / 'baseline-a-inference-manifest.txt').write_bytes(raw)
    manifest = json.loads(raw)
    assert manifest['origin']['kind'] == 'all-neural-weights-random'
    publication.append({'product': 'A-moe-joint', 'url': a_url, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'accessed_at': datetime.now(timezone.utc).isoformat(), 'origin': manifest['origin'],
                        'payload_files': list(manifest['files']), 'weights_downloaded': False, 'inference_run': False})
    results['public_baseline_metadata'] = publication
    source_paths = ['tiny_perceptron/training.py','tiny_perceptron/multimodal.py','tiny_perceptron/omni_products/av.py',
                    'tiny_perceptron/lesson_products/release.json','tiny_perceptron/selftrained/inference.py',
                    'tiny_perceptron/selftrained/dataset.py','tiny_perceptron/lesson_products/delivery.py',
                    'tiny_perceptron/omni_products/routing.py','tiny_perceptron/omni_products/questions.py',
                    'scripts/train.py','scripts/train_simple.py']
    results['additional_source_hashes'] = []
    for name in source_paths:
        raw=(ROOT/name).read_bytes()
        results['additional_source_hashes'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        if name.endswith('.py'): ast.parse(raw)
    code = (ROOT/'tiny_perceptron/omni_products/training_runtime.py').read_text()
    assert 'from_pretrained' not in code and 'transformers' not in code
    assert 'model = AVModel' in code and 'model = ChoiceModel' in code and 'model = SoundCodec' in code
    results['new_training_source_route'] = {'checked': 'train_av/train_choice/train_codec/train_talker construct local models',
                                           'mature_model_import_or_from_pretrained': False,
                                           'scope': 'source provenance of the declared current training entrypoints; no training executed'}
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_glossary_whole',
            'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/omni-P5-glossary/supplement-checks-v1.py',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU'},'status':'passed','checks':results}
    path=PROOF/'supplement-checks-v1.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','output':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
