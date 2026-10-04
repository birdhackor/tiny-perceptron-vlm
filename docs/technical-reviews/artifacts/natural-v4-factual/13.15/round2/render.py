"""Render original prerequisite SVGs through installed librsvg/cairo, without installation."""
import ctypes as C
import ctypes.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
rsvg_path = ctypes.util.find_library("rsvg-2")
cairo_path = ctypes.util.find_library("cairo")
rsvg = C.CDLL(rsvg_path)
cairo = C.CDLL(cairo_path)
gobject = C.CDLL(ctypes.util.find_library("gobject-2.0"))


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
cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
cairo.cairo_surface_write_to_png.restype = C.c_int
cairo.cairo_destroy.argtypes = [C.c_void_p]
cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
gobject.g_object_unref.argtypes = [C.c_void_p]

records = []
for name in ["posttrain_stages"]:
    svg = ROOT / "course/figures" / (name + ".svg")
    png = OUT / (name + ".png")
    handle = rsvg.rsvg_handle_new_from_file(str(svg).encode(), None)
    assert handle
    dims = Dimensions()
    rsvg.rsvg_handle_get_dimensions(handle, C.byref(dims))
    surface = cairo.cairo_image_surface_create(0, dims.width, dims.height)
    context = cairo.cairo_create(surface)
    rendered = rsvg.rsvg_handle_render_cairo(handle, context)
    status = cairo.cairo_surface_write_to_png(surface, str(png).encode())
    assert rendered == 1 and status == 0
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)
    records.append({"source": str(svg.relative_to(ROOT)), "source_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(), "render": str(png.relative_to(ROOT)), "size": [dims.width, dims.height], "librsvg_render_success": rendered, "cairo_png_status": status})
record = {"command": ".venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/13.15/round2/render.py", "environment": {"python": "3.13.5", "rsvg_library": str(rsvg_path), "cairo_library": str(cairo_path), "device": "cpu"}, "renders": records}
(OUT / "render-execution.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
