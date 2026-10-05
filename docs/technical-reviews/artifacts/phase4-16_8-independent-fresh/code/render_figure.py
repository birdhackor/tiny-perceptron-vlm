import hashlib,json,subprocess
from pathlib import Path
from PIL import Image
ROOT=Path('/workspace/tiny-perceptron-vlm');OUT=ROOT/'docs/technical-reviews/artifacts/phase4-16_8-independent-fresh'
svg=OUT/'inputs/course__figures__rewrite-16-causal-mask.svg'
png=OUT/'figure.png'
previous=hashlib.sha256(png.read_bytes()).hexdigest()
command=['inkscape',str(svg.relative_to(ROOT)),'--export-type=png','--export-filename='+str(png.relative_to(ROOT)),'--export-width=1320']
result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
(OUT/'figure_render_stdout.txt').write_text(result.stdout)
(OUT/'figure_render_stderr.txt').write_text(result.stderr)
current=hashlib.sha256(png.read_bytes()).hexdigest()
assert current==previous
log={'command':command,'exit_code':result.returncode,'inkscape_version':subprocess.run(['inkscape','--version'],capture_output=True,text=True,check=True).stdout.strip(),'svg_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':current,'png_pixels':list(Image.open(png).size),'matches_previously_viewed_render_sha':current==previous,'actual_visual_inspection':'Reviewer invoked view_image and inspected this PNG: rows Q0-Q3 and columns K positions0-3; ten green1 cells on/below diagonal; six gray0 future cells; Q0 reads0, Q3 reads0-3; no arrows; legend1允許/0禁止 and labels agree with prose/torch.tril.','warnings':'Inkscape PangoFT2FontMap and GtkRecentManager wrap warnings; completed successfully and rendered Chinese labels clearly.'}
(OUT/'figure_render.json').write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(log,ensure_ascii=False,indent=2))
