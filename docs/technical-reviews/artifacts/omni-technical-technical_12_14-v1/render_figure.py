"""用既有 librsvg/Cairo C API 原尺寸渲染 SVG；PNG 保存在 Git 外。"""

import ctypes as c
import ctypes.util as u
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
R = c.CDLL(u.find_library('rsvg-2'))
C = c.CDLL(u.find_library('cairo'))
G = c.CDLL(u.find_library('gobject-2.0'))
R.rsvg_handle_new_from_file.argtypes = [c.c_char_p, c.c_void_p]
R.rsvg_handle_new_from_file.restype = c.c_void_p
C.cairo_image_surface_create.argtypes = [c.c_int, c.c_int, c.c_int]
C.cairo_image_surface_create.restype = c.c_void_p
C.cairo_create.argtypes = [c.c_void_p]
C.cairo_create.restype = c.c_void_p
R.rsvg_handle_render_cairo.argtypes = [c.c_void_p, c.c_void_p]
R.rsvg_handle_render_cairo.restype = c.c_int
C.cairo_surface_write_to_png.argtypes = [c.c_void_p, c.c_char_p]
C.cairo_surface_write_to_png.restype = c.c_int
C.cairo_destroy.argtypes = [c.c_void_p]
C.cairo_surface_destroy.argtypes = [c.c_void_p]
G.g_object_unref.argtypes = [c.c_void_p]
C.cairo_version_string.restype = c.c_char_p
src = ROOT / 'course/figures/rewrite-12-asr-history.svg'
dst = Path('/workspace/work/reference/omni-technical-technical_12_14-v1/rewrite-12-asr-history-final.png')
assert not dst.exists()
handle = R.rsvg_handle_new_from_file(str(src).encode(), None)
assert handle
surface = C.cairo_image_surface_create(0, 640, 725)
ctx = C.cairo_create(surface)
assert R.rsvg_handle_render_cairo(handle, ctx) == 1
assert C.cairo_surface_write_to_png(surface, str(dst).encode()) == 0
C.cairo_destroy(ctx)
C.cairo_surface_destroy(surface)
G.g_object_unref(handle)
print(json.dumps({
    'rendered_at': datetime.now(UTC).isoformat(),
    'source': src.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
    'render': str(dst), 'render_sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
    'render_bytes': dst.stat().st_size, 'width': 640, 'height': 725,
    'renderer': {'librsvg': '.'.join(str(c.c_uint.in_dll(R, f'rsvg_{part}_version').value) for part in ['major', 'minor', 'micro']),
                 'cairo': C.cairo_version_string().decode(), 'device': 'cpu'},
    'method': 'rsvg_handle_new_from_file -> rsvg_handle_render_cairo -> cairo_surface_write_to_png; Noto Sans CJK TC system font',
    'prior_attempts': 'Chromium screenshots timed out (v1 terminated; isolated v2 timeout exit 124); ImageMagick exit 1 because rsvg-convert executable absent. Final render uses installed shared libraries without new dependencies.',
}, ensure_ascii=False, indent=2))
