"""Bounded primary-source extraction and figure rendering; no model work."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]

def run(argv, label):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        (BASE/'execution'/f'{label}.stdout.txt').write_bytes(error.stdout or b'')
        (BASE/'execution'/f'{label}.stderr.txt').write_bytes(error.stderr or b'')
        (BASE/'execution'/f'{label}.json').write_text(json.dumps({
            'argv':argv, 'exit_code':124, 'timeout_seconds':30,
            'device':'CPU', 'python':sys.version, 'limitation':'actual browser timeout; no screenshot inspection claimed'},indent=2)+'\n')
        raise
    (BASE / 'execution' / f'{label}.stdout.txt').write_text(result.stdout)
    (BASE / 'execution' / f'{label}.stderr.txt').write_text(result.stderr)
    receipt = {'argv': argv, 'cwd': str(Path.cwd()), 'exit_code': result.returncode,
               'python': sys.version, 'device': 'CPU',
               'stdout_file': f'execution/{label}.stdout.txt',
               'stderr_file': f'execution/{label}.stderr.txt'}
    (BASE / 'execution' / f'{label}.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed: {result.returncode}')

run(['/usr/bin/pdftotext', '-layout', str(BASE/'sources/position-interpolation-2306.15595v2.pdf'),
     str(BASE/'sources/position-interpolation-2306.15595v2.txt')], 'pdftotext')
for svg in (BASE / 'inputs').glob('*.svg'):
    run(['/usr/bin/inkscape', str(svg), '--export-type=png',
         f'--export-filename={BASE / "renders" / (svg.stem+".png")}'], 'render-'+svg.stem)
html = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>14.8 figure inspection</title>
<style>body{margin:0;padding:24px;font-family:sans-serif;background:white}main{max-width:640px;margin:auto}img{width:100%;height:auto;display:block}p{font-size:18px;line-height:1.6}</style>
<main><p>十六張卡片全部保留，位置0到15分別映成0到7.5</p><img src="../inputs/rewrite-14-8-position-interpolation.svg" alt="位置插值"><p>這是圖與圖說的響應式檢查，不是完整課程網站驗收。</p></main></html>'''
(BASE/'renders/figure-context.html').write_text(html)
for name, size in [('desktop', '1280,900'), ('mobile', '390,844')]:
    run(['/usr/bin/chromium', '--headless', '--no-sandbox', '--disable-gpu',
         '--disable-dev-shm-usage', '--disable-background-networking', '--no-first-run',
         '--disable-extensions', '--disable-component-update',
         f'--user-data-dir=/tmp/phase4-14_8-independent-{name}-chromium',
         '--hide-scrollbars', f'--window-size={size}',
         f'--screenshot={BASE / "renders" / ("14_8-"+name+".png")}',
         (BASE/'renders/figure-context.html').as_uri()], 'chromium-'+name)
print(json.dumps({'pdf_extracted':True,'svg_renders':4,'responsive_figure_screenshots':2}))
