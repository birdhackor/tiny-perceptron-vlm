import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources = BASE/'sources'
assert (sources/'cache-explanation-v4.57.1.md').read_bytes() == (sources/'cache-explanation-tag-v4.57.1.md').read_bytes()
assert (sources/'vllm-metrics-v0.10.2.py').read_bytes() == (sources/'vllm-metrics-01efc7e.py').read_bytes()
assert '8cb5963cc22174954e7dca2c0a3320b7dc2f4edc\trefs/tags/v4.57.1^{}' in (sources/'transformers-tag-verification.txt').read_text()
assert '01efc7ef781391e744ed08c3292817a773d654e6\trefs/tags/v0.10.2' in (sources/'vllm-tag-verification.txt').read_text()
assert 'arXiv:2408.03314v1 [cs.LG] 6 Aug 2024' in (sources/'snell-2408.03314v1.txt').read_text()
records = [
    {'file':'snell-2408.03314v1.pdf','url':'https://arxiv.org/pdf/2408.03314v1','version':'arXiv:2408.03314v1, submitted 6 Aug 2024; PDF header 2024-8-7',
     'command':'curl --fail --location --max-time 60 https://arxiv.org/pdf/2408.03314v1 --output docs/technical-reviews/artifacts/phase4-c_5-independent-20261005/sources/snell-2408.03314v1.pdf',
     'authority':'Original author paper, hosted by arXiv; personally verified authors/version on the PDF first page.',
     'read_locators':'Front page; §1 pp.1–3; §2 pp.3–4; §3/§3.1 Eq.(1) pp.4–5; §5.1/5.2 pp.6–7'},
    {'file':'cache-explanation-v4.57.1.md','url':'https://raw.githubusercontent.com/huggingface/transformers/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/docs/source/en/cache_explanation.md','version':'Transformers v4.57.1, dereferenced official tag commit 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc',
     'command':'curl --fail --location --max-time 45 https://raw.githubusercontent.com/huggingface/transformers/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/docs/source/en/cache_explanation.md --output docs/technical-reviews/artifacts/phase4-c_5-independent-20261005/sources/cache-explanation-v4.57.1.md',
     'authority':'Hugging Face official Transformers repository; official tag dereference and independent tag/commit byte equality checked.',
     'read_locators':'Lines 16–84, Caching introduction, Attention matrices and Cache class'},
    {'file':'vllm-metrics-01efc7e.py','url':'https://raw.githubusercontent.com/vllm-project/vllm/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/metrics/loggers.py','version':'vLLM v0.10.2, official tag commit 01efc7ef781391e744ed08c3292817a773d654e6',
     'command':'curl --fail --location --max-time 45 https://raw.githubusercontent.com/vllm-project/vllm/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/metrics/loggers.py --output docs/technical-reviews/artifacts/phase4-c_5-independent-20261005/sources/vllm-metrics-01efc7e.py',
     'authority':'vLLM project official repository; official tag and independent tag/commit byte equality checked.',
     'read_locators':'Lines 410–452 metric definitions and 560–574 observed timing categories'}]
for r in records:
    r.update(accessed_on='2026-10-05',sha256=sha(sources/r['file']),tls_verification='Enabled; curl --fail, no insecure options')
(sources/'authority-receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'original_sources':len(records),'version_and_byte_checks':'all passed'},ensure_ascii=False))
