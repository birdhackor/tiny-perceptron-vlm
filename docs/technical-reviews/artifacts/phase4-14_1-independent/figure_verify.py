"""Check rendered SVG geometry independently of its printed labels."""
from pathlib import Path
import xml.etree.ElementTree as ET
import math
import json

BASE=Path('docs/technical-reviews/artifacts/phase4-14_1-independent')
ns={'s':'http://www.w3.org/2000/svg'}
a=ET.parse('course/figures/rewrite-14-shared-rotation.svg').getroot()
found={}
for color,key in [('#107b6e','Q'),('#ac5016','K')]:
    z=[n for n in a.findall('s:line',ns) if n.get('stroke')==color]
    found[key]=[(float(n.get('x2'))-float(n.get('x1')),
                 -(float(n.get('y2'))-float(n.get('y1')))) for n in z]
for i in [0,1]:
    q,k=found['Q'][i],found['K'][i]
    dot=sum(x*y for x,y in zip(q,k))/(math.hypot(*q)*math.hypot(*k))
    angle=math.degrees(math.acos(dot))
    print('shared frame',i,'math vectors',q,k,'unit dot',dot,'angle',angle)
    assert abs(angle-135)<1e-9 and abs(dot-math.cos(3*math.pi/4))<1e-12
b=ET.parse('course/figures/rewrite-14-rotation-components.svg').getroot()
circles=[n for n in b.findall('s:circle',ns) if n.get('r')=='93']
arrows=[n for n in b.findall('s:line',ns) if n.get('stroke')=='#245ba8']
coords=[]
for circle,arrow,sign in zip(circles,arrows,[1,-1]):
    dx=float(arrow.get('x2'))-float(arrow.get('x1'))
    dy=-(float(arrow.get('y2'))-float(arrow.get('y1')))
    r=float(circle.get('r'))
    x,y=dx/r,dy/r
    coords.append([x,y])
    assert abs(x-sign/math.sqrt(2))<1e-12 and abs(y-1/math.sqrt(2))<1e-12
print('component vectors/radius',coords,'circle_dasharray',[n.get('stroke-dasharray') for n in circles])
(BASE/'figure-geometry.json').write_text(json.dumps({
    'shared_vectors_math_y_up':found,'component_unit_vectors':coords,
    'component_circle_dasharray':[n.get('stroke-dasharray') for n in circles]},indent=2)+'\n')
