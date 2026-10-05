"""Render the exact frozen W.6 SVG and independently check its coordinate map."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright

BASE=Path(__file__).resolve().parent
svg_path=BASE/'frozen-input/course/figures/rewrite-W-6-gradient-step.svg'
raw=svg_path.read_bytes()
root=ET.fromstring(raw)
ns={'s':'http://www.w3.org/2000/svg'}
circles=root.findall('s:circle',ns)
expected=[(1,4),(1.4,2.56),(3,0)]
points=[]
for circle,(w,L) in zip(circles,expected,strict=True):
    x,y=float(circle.attrib['cx']),float(circle.attrib['cy'])
    assert abs(x-(75+123*w))<1e-9 and abs(y-(430-34*L))<1e-9
    points.append({'w':w,'L':L,'svg_x':x,'svg_y':y})
curve=next(p for p in root.findall('s:path',ns) if p.attrib.get('stroke-width')=='3' and p.attrib.get('stroke')=='#64748b')
vertices=[(float(x),float(y)) for x,y in re.findall(r'[ML]([\d.]+) ([\d.]+)',curve.attrib['d'])]
residuals=[abs(y-(430-34*((x-75)/123-3)**2)) for x,y in vertices]
assert max(residuals)<0.03
arrow=next(p for p in root.findall('s:path',ns) if p.attrib.get('marker-end'))
arrow_points=[(float(x),float(y)) for x,y in re.findall(r'[ML]([\d.]+) ([\d.]+)',arrow.attrib['d'])]
assert arrow_points[1][0]>arrow_points[0][0] and arrow_points[1][1]>arrow_points[0][1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':640,'height':555},device_scale_factor=1)
    page.set_content('<style>body{margin:0}</style>'+raw.decode('utf-8'),wait_until='load')
    page.screenshot(path=str(BASE/'figure-640x555.png'),full_page=True)
    version=browser.version
    browser.close()
result={'svg_sha256':hashlib.sha256(raw).hexdigest(),'coordinate_map':'x=75+123*w, y=430-34*L; pixel y increases downwards','points':points,'curve_vertices_checked':len(vertices),'max_curve_pixel_residual':max(residuals),'curve_tolerance_pixels':0.03,'arrow_svg_points':arrow_points,'arrow_interpretation':'Result-to-result connector with increasing w and decreasing L, not tangent/derivative','browser':'Chromium '+version,'render':'figure-640x555.png'}
(BASE/'figure-check-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
