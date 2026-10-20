import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skipping += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skipping = max(0, self.skipping - 1)

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


TORCH = 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/'
HF = 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/'
JOBS = [
    ('cache', HF+'docs/source/en/cache_explanation.md', 'Transformers v4.57.1', False,
     [('Cache reuse', 'A key-value (KV) cache', 400),
      ('Cache position', 'cache_position', 700), ('Attention mask', 'attention mask', 700)]),
    ('llama', HF+'src/transformers/models/llama/modeling_llama.py', 'Transformers v4.57.1', False,
     [('RMSNorm', 'class LlamaRMSNorm', 1000),
      ('RoPE', 'def apply_rotary_pos_emb', 1900),
      ('GQA', 'def repeat_kv', 850), ('Weight tying', '_tied_weights_keys', 350)]),
    ('mixtral', HF+'src/transformers/models/mixtral/modeling_mixtral.py', 'Transformers v4.57.1', False,
     [('Top-k routing', 'routing_weights, selected_experts', 1800)]),
    ('tying', HF+'src/transformers/modeling_utils.py', 'Transformers v4.57.1', False,
     [('Shared embedding/output parameter', '    def _tie_or_clone_weights', 1150)]),
    ('sdpa', TORCH+'torch/nn/functional.py', 'PyTorch 2.14.1 git 5c4886908584029761b579af026dcfb627c84070', False,
     [('Scaled dot product attention', 'scaled_dot_product_attention =', 1000),
      ('Backend selection', 'There are currently three', 2100),
      ('Floating point differences', 'Due to the nature of fusing', 600)]),
    ('grad', TORCH+'torch/autograd/grad_mode.py', 'PyTorch 2.14.1 git 5c4886908584029761b579af026dcfb627c84070', False,
     [('no_grad', 'class no_grad', 800), ('Gradient state', 'def __enter__', 400)]),
    ('module', TORCH+'torch/nn/modules/module.py', 'PyTorch 2.14.1 git 5c4886908584029761b579af026dcfb627c84070', False,
     [('eval', '    def eval(', 1100)]),
    ('gqa', 'https://arxiv.org/html/2305.13245v3', 'arXiv 2305.13245v3, 2023-12-23', True,
     [('Grouped-query attention', '2.2 Grouped-query attention', 2000)]),
    ('packing', 'https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html',
     'Megatron Bridge 0.6.2, redirected documentation, retrieved snapshot', True,
     [('Packing definition', 'Packed sequences are a fine-tuning technique', 900)]),
    ('speculative', 'https://arxiv.org/html/2211.17192v2', 'arXiv 2211.17192v2, 2023-05-18', True,
     [('Draft and verification', 'Algorithm 1 SpeculativeDecodingStep', 2100),
      ('No retraining', 'without retraining', 350)]),
    ('mtp', 'https://arxiv.org/html/2412.19437v2', 'DeepSeek-V3 Technical Report, arXiv 2412.19437v2', True,
     [('MTP in Inference', 'MTP in Inference.', 450)]),
    ('historical_dataset', 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/3f9fbb2a04f08034c0b630e837e7afba1f584402/tiny_perceptron/selftrained/dataset.py',
     'training_revision 3f9fbb2a04f08034c0b630e837e7afba1f584402', False,
     [('One record per row', '    def batch(', 1400)]),
    ('historical_train', 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/3f9fbb2a04f08034c0b630e837e7afba1f584402/scripts/selftrained/train.py',
     'training_revision 3f9fbb2a04f08034c0b630e837e7afba1f584402', False,
     [('Training objective batching', 'def objective(', 700)]),
    ('historical_model', 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/bf43be260d3e819f4ff3df3b1155daac98d01071/tiny_perceptron/selftrained/model.py',
     'final_test.revision bf43be260d3e819f4ff3df3b1155daac98d01071', False,
     [('Generation implementation', '    def generate(', 2200)]),
]


def fetch(job):
    identifier, url, version, html, selections = job
    command = ['curl', '-fLsS', '--max-time', '40', '-w', '\n%{url_effective}', url]
    run = subprocess.run(command, capture_output=True, check=True)
    raw, _, final_url = run.stdout.rpartition(b'\n')
    content = raw.decode()
    if html:
        parser = PlainText()
        parser.feed(content)
        content = ' '.join(' '.join(parser.parts).split())
    excerpts = []
    for locator, needle, length in selections:
        start = (content.lower().rfind(needle.lower()) if identifier in ('gqa', 'mtp')
                 else content.lower().find(needle.lower()))
        if start < 0:
            excerpts.append({'locator': locator, 'found': False, 'needle': needle})
        else:
            excerpts.append({'locator': locator, 'found': True, 'text': content[start:start+length]})
    result = {'id': identifier, 'url': url, 'final_url': final_url.decode(), 'version': version,
            'accessed_on': date.today().isoformat(), 'full_response_sha256': hashlib.sha256(raw).hexdigest(),
            'full_response_bytes': len(raw), 'command': command, 'exit_code': run.returncode,
            'full_text_saved': False, 'excerpts': excerpts}
    local_paths = {'historical_dataset': 'tiny_perceptron/selftrained/dataset.py',
                   'historical_train': 'scripts/selftrained/train.py',
                   'historical_model': 'tiny_perceptron/selftrained/model.py'}
    if identifier in local_paths:
        local = Path(__file__).resolve().parents[4] / local_paths[identifier]
        result['local_path'] = local_paths[identifier]
        result['local_sha256'] = hashlib.sha256(local.read_bytes()).hexdigest()
        result['identical_to_local'] = raw == local.read_bytes()
    return result


with ThreadPoolExecutor(max_workers=5) as executor:
    sources = list(executor.map(fetch, JOBS))
print(json.dumps({'sources': sources}, ensure_ascii=False, indent=2))
