"""Select exact original text/HTML excerpts; record full downloaded-source hashes."""
import hashlib
import json
from pathlib import Path

originals = Path('/tmp/phase4-16_12-original-sources')
destination = Path(__file__).resolve().parent
acquisition = json.loads((originals / 'download.json').read_text())
records = []


def save(name, source, selected, locator, origin_file):
    raw = (originals / source).read_bytes()
    assert selected in raw
    (destination / name).write_bytes(selected)
    assert hashlib.sha256((destination / name).read_bytes()).digest() == hashlib.sha256(selected).digest()
    origin = next(r for r in acquisition if r['file'] == origin_file)
    assert hashlib.sha256((originals / origin_file).read_bytes()).hexdigest() == origin['sha256']
    records.append({'snapshot': name, 'snapshot_sha256': hashlib.sha256(selected).hexdigest(), 'selected_source': source, 'selected_source_sha256': hashlib.sha256(raw).hexdigest(), 'selection_locator': locator, 'original_source': origin, 'preserved_scope': 'Exact selected bytes of the original HTML or pdftotext output; full original document not copied.'})


raw = (originals / 'mistral-pages-2-3.txt').read_bytes()
start = raw.index(b'2       Architectural details')
end = raw.index(b'3       Results')
save('mistral-section2.txt', 'mistral-pages-2-3.txt', raw[start:end], 'PDF pp.2-3, section 2 including figures 1-3; ends before section 3', 'mistral-v1.pdf')

raw = (originals / 'attention-pages-3-4.txt').read_bytes()
start = raw.index(b'3.1   Encoder and Decoder Stacks')
end = raw.index(b'3.2.2   Multi-Head Attention')
save('attention-sections3-1-3-2-1.txt', 'attention-pages-3-4.txt', raw[start:end], 'PDF pp.3-4, sections 3.1, 3.2, 3.2.1 and equation (1)', 'attention-v7.pdf')
raw = (originals / 'attention-page-6.txt').read_bytes()
save('attention-table1.txt', 'attention-page-6.txt', raw[:raw.index(b'3.5    Positional Encoding')], 'PDF p.6 table 1; dense and restricted attention complexity rows', 'attention-v7.pdf')

raw = (originals / 'rag-pages-1-3.txt').read_bytes()
start = raw.index(b'2      Methods')
end = raw.index(b'To train the retriever and generator end-to-end')
save('rag-section2-opening.txt', 'rag-pages-1-3.txt', raw[start:end], 'PDF pp.2-3 section 2 opening; retriever plus generator using passages as context', 'rag-v4.pdf')

raw = (originals / 'gemma3-page2-reading-order.txt').read_bytes()
start = raw.index(b'5:1 interleaving of local/global layers.')
end = raw.index(b'Vision encoder.', start)
save('gemma3-local-global-start.txt', 'gemma3-page2-reading-order.txt', raw[start:end], 'PDF p.2 left column section 2, 5:1 interleaving paragraph beginning', 'gemma3-v1.pdf')
start = raw.index(b'attention (Luong et al., 2015), with a pattern of')
end = raw.index(b'Long context.', start)
save('gemma3-local-global-continuation.txt', 'gemma3-page2-reading-order.txt', raw[start:end], 'PDF p.2 right column, continuation of 5:1 interleaving paragraph', 'gemma3-v1.pdf')

selections = {
    'stdtypes': [(1495,1530),(4585,4620),(4680,4689)],
    'functions': [(1354,1364),(1433,1453),(2077,2087),(2155,2162)],
    'expressions': [(483,503),(1025,1033)],
}
for page, ranges in selections.items():
    source = f'python-3.13.5-{page}.html'
    lines = (originals / source).read_bytes().splitlines(keepends=True)
    for lo, hi in ranges:
        save(f'python-{page}-lines-{lo}-{hi}.html', source, b''.join(lines[lo-1:hi]), f'Original downloaded HTML lines {lo}-{hi}', source)

(destination / 'provenance.json').write_text(json.dumps({'accessed_on': '2026-10-05', 'acquisition_attempts': acquisition, 'snapshots': records, 'authority_scope': 'Original papers and official Python docs were personally inspected. Excerpts support method definitions only; their model results were not re-evaluated.'}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'snapshots': len(records), 'full_original_sha_checks': len(records), 'copy_sha_checks': len(records)}, indent=2))
