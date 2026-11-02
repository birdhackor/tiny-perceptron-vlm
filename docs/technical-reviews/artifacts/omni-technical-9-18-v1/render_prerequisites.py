"""Render the two explicitly linked 9.10 prerequisite SVGs with installed libraries."""

import ctypes as C
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-9-18-v1")
rsvg = C.CDLL("librsvg-2.so.2")
cairo = C.CDLL("libcairo.so.2")
gobject = C.CDLL("libgobject-2.0.so.0")


class Dimensions(C.Structure):
    _fields_ = [("width", C.c_int), ("height", C.c_int), ("em", C.c_double), ("ex", C.c_double)]


rsvg.rsvg_handle_new_from_file.argtypes = [C.c_char_p, C.c_void_p]
rsvg.rsvg_handle_new_from_file.restype = C.c_void_p
rsvg.rsvg_handle_get_dimensions.argtypes = [C.c_void_p, C.POINTER(Dimensions)]
rsvg.rsvg_handle_render_cairo.argtypes = [C.c_void_p, C.c_void_p]
rsvg.rsvg_handle_render_cairo.restype = C.c_int
cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
cairo.cairo_image_surface_create.restype = C.c_void_p
cairo.cairo_create.argtypes = [C.c_void_p]
cairo.cairo_create.restype = C.c_void_p
cairo.cairo_set_source_rgb.argtypes = [C.c_void_p, C.c_double, C.c_double, C.c_double]
cairo.cairo_paint.argtypes = [C.c_void_p]
cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
cairo.cairo_surface_write_to_png.restype = C.c_int
cairo.cairo_destroy.argtypes = [C.c_void_p]
cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
gobject.g_object_unref.argtypes = [C.c_void_p]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


entries = []
for filename, output_name in (
    ("rewrite-09-10-calibration-bins.svg", "prerequisite-9-10-bins.png"),
    ("audio_calibration_test.svg", "prerequisite-9-10-audio.png"),
):
    source = ROOT / "course/figures" / filename
    output = CACHE / output_name
    handle = rsvg.rsvg_handle_new_from_file(str(source).encode(), None)
    assert handle
    dimensions = Dimensions()
    rsvg.rsvg_handle_get_dimensions(handle, C.byref(dimensions))
    surface = cairo.cairo_image_surface_create(0, dimensions.width, dimensions.height)
    context = cairo.cairo_create(surface)
    cairo.cairo_set_source_rgb(context, 1, 1, 1)
    cairo.cairo_paint(context)
    assert rsvg.rsvg_handle_render_cairo(handle, context)
    assert cairo.cairo_surface_write_to_png(surface, str(output).encode()) == 0
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)
    entries.append({"svg": source.relative_to(ROOT).as_posix(), "svg_sha256": sha(source), "png_git_external": str(output), "png_sha256": sha(output), "pixels": [dimensions.width, dimensions.height]})

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-18-v1/render_prerequisites.py",
    "environment": {"librsvg2": "2.60.0+dfsg-1", "libcairo2": "1.18.4-1+b1", "device": "CPU"},
    "method": "ctypes calls installed librsvg rsvg_handle_render_cairo on a white Cairo ARGB32 surface; SVG intrinsic dimensions; PNGs stay outside Git.",
    "rendered": entries,
    "scope": "9.18 has no figure. These are only its explicitly linked 9.10 prerequisite figures; the 9.18 figure_sha256 map stays empty.",
}
out = ROOT / "docs/technical-reviews/artifacts/omni-technical-9-18-v1/render-receipt.json"
with out.open("x") as file:
    json.dump(receipt, file, ensure_ascii=False, indent=2)
    file.write("\n")
print(json.dumps({"receipt": out.relative_to(ROOT).as_posix(), "sha256": sha(out), "rendered": entries}, ensure_ascii=False))
