"""Execute the frozen original and requested exercise; independently check rational coordinates."""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def execute(path, name):
    argv = [sys.executable, '-I', str(path)]
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=10)
    (BASE/'execution'/f'{name}.stdout.txt').write_text(result.stdout)
    (BASE/'execution'/f'{name}.stderr.txt').write_text(result.stderr)
    receipt = {'command_argv':argv, 'cwd':str(ROOT), 'exit_code':result.returncode,
               'code_sha256':digest(path.read_bytes()), 'python':sys.version,
               'executable':sys.executable, 'device':'CPU', 'platform':platform.platform(),
               'stdout_sha256':digest(result.stdout.encode()), 'stderr_sha256':digest(result.stderr.encode())}
    (BASE/'execution'/f'{name}.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    assert result.returncode == 0
    return result.stdout

original = BASE/'inputs/fence-1.py'
original_stdout = execute(original, 'original-fence')
expected = '卡片數 16 座標數 16\n0 → 0.0\n1 → 0.5\n2 → 1.0\n14 → 7.0\n15 → 7.5\n'
assert original_stdout == expected
namespace = {}
# Literal AST inspection verifies that the original is a coordinate-only calculation.
tree = ast.parse(original.read_bytes())
assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) for n in ast.walk(tree))
calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert set(calls) == {'list','range','print','len'}

variant = original.read_text().replace('8, 16','8, 24').replace('(0, 1, 2, 14, 15)','(5, 23)')
variant_path = BASE/'code/exercise-24.py'
variant_path.write_text(variant)
exercise_stdout = execute(variant_path, 'exercise-24')
assert exercise_stdout == '卡片數 24 座標數 24\n5 → 1.6666666666666667\n23 → 7.666666666666667\n'

checks = []
for new_length in (8,16,24):
    # Fraction arithmetic provides an exact, separate reference for count, spacing and bounds.
    expected_coordinates = [Fraction(m * 8, new_length) for m in range(new_length)]
    computed = [m * 8 / new_length for m in range(new_length)]
    assert len(computed) == new_length and len(set(computed)) == new_length
    max_error = max(abs(float(a)-b) for a,b in zip(expected_coordinates,computed))
    assert max_error <= 2e-15
    assert all(0 <= x < 8 for x in expected_coordinates)
    assert expected_coordinates[-1] == 8-Fraction(8,new_length)
    assert all(b-a == Fraction(8,new_length) for a,b in zip(expected_coordinates,expected_coordinates[1:]))
    checks.append({'new_length':new_length, 'coordinate_count':len(computed),
                   'distinct_coordinates':len(set(computed)),
                   'last_coordinate_exact':str(expected_coordinates[-1]),
                   'step_exact':str(Fraction(8,new_length)), 'max_float_error':max_error})

# Toy angle explicitly uses degrees, converted to radians only for a real rotation.
before_degrees = 2*60
after_degrees = float(Fraction(2*8,16))*60
assert (before_degrees,after_degrees)==(120,60)
original_gap = 15-0
scaled_gap = Fraction(15*8,16)-Fraction(0*8,16)
assert (original_gap,scaled_gap)==(15,Fraction(15,2))
adjacent_before = 60
adjacent_after = float(Fraction(8,16))*60
assert (adjacent_before,adjacent_after)==(60,30)
unit_arrow = [math.cos(math.radians(after_degrees)),math.sin(math.radians(after_degrees))]
assert abs(unit_arrow[0]-0.5)<1e-15
assert abs(unit_arrow[1]-math.sqrt(3)/2)<1e-15

# Match every card label in the SVG, not just the five printed examples.
svg = ET.parse(BASE/'inputs/rewrite-14-8-position-interpolation.svg')
ns = {'s':'http://www.w3.org/2000/svg'}
all_text = [(el.attrib, ''.join(el.itertext())) for el in svg.findall('.//s:text',ns)]
token_text = [text for attrs,text in all_text if attrs.get('font-size')=='32']
positions = [int(text) for attrs,text in all_text if attrs.get('y') in {'187','391'}]
coordinates = [float(text) for attrs,text in all_text if attrs.get('y') in {'263','467'}]
assert ''.join(token_text) == '校慶週六上午開始請在操場入口集合'
assert len(token_text)==16
assert positions==list(range(16))
assert coordinates==[float(Fraction(m,2)) for m in positions]
arrows = svg.findall('.//s:line',ns)
assert len(arrows)==16 and all(x.attrib.get('marker-end')=='url(#arrow)' for x in arrows)
assert all(float(x.attrib['y2'])>float(x.attrib['y1']) for x in arrows)

measurement = {'checks':checks,'angles':{'units':'degrees; radian conversion used for cos/sin only',
               'position_2_before':before_degrees,'position_2_after':after_degrees,
               'adjacent_angle_before':adjacent_before,'adjacent_angle_after':adjacent_after},
               'distance':{'units':'position coordinate units','unscaled_0_to_15':original_gap,'scaled_0_to_15':float(scaled_gap)},
               'exercise_rounded_4_decimals':[round(5/3,4),round(23/3,4)],
               'figure':{'cards':len(token_text),'positions':positions,'coordinates':coordinates,'downward_arrows':len(arrows)},
               'original_ast_call_names':calls,'scope':'coordinate and toy rotation arithmetic only; no training, gradient, model loading or evaluation',
               'tolerance':'half-grid exact; rational-to-float error <=2e-15; four-decimal exercises <=0.00005'}
(BASE/'execution/measurements.json').write_text(json.dumps(measurement,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(measurement,ensure_ascii=False))
