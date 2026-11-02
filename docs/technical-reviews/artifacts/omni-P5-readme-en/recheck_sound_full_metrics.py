"""Pure replay of all sound acceptance conditions from preserved portable event rows."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

directory = Path(__file__).resolve().parent
source = directory / 'new-results-recomputed-v2.json'
raw = source.read_bytes()
rows = json.loads(raw)['sound']['rows']
successes = []
for row in rows:
    actual, expected = row['measured'], row['expected']
    ok = row['eos'] and not row['capped'] and row['legal'] and len(actual) == len(expected)
    for x, y in zip(actual, expected):
        ok &= abs(x['frequency'] - y['frequency']) / y['frequency'] <= 0.05
        ok &= abs(x['onset_ms'] - y['onset_ms']) <= 30
        ok &= abs(x['offset_ms'] - y['offset_ms']) <= 30
        ok &= abs((x['offset_ms'] - x['onset_ms']) - (y['offset_ms'] - y['onset_ms'])) <= 30
    for x, following, y, next_target in zip(actual, actual[1:], expected, expected[1:]):
        ok &= abs((following['onset_ms'] - x['offset_ms']) - (next_target['onset_ms'] - y['offset_ms'])) <= 30
    assert ok == row['success_recomputed']
    successes.append(bool(ok))
result = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'input_path': source.name, 'input_sha256': hashlib.sha256(raw).hexdigest(),
    'source_contract': 'tiny_perceptron/omni_products/signal_metrics.py:88-110 program_success and core_evaluation.generated_sound EOS/capped conditions',
    'conditions_checked': ['EOS', 'not capped', 'legal signal', 'event count', 'ordered frequencies <=5%', 'onsets <=30ms', 'offsets <=30ms', 'durations <=30ms', 'inter-event rests <=30ms'],
    'families': len(successes), 'successful': sum(successes),
    'result': 'All 200 saved held-out programs meet every frozen condition; original numerical event analyses replayed without waveform or neural remeasurement.'
}
assert result['families'] == result['successful'] == 200
output = directory / 'sound-full-metrics-v1.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
