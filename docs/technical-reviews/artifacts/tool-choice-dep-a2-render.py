"""Render the actual A.2 SVG using installed librsvg/Cairo, without a browser."""
import ctypes as C
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
source = root / "course/figures/rag.svg"
output = root / "docs/technical-reviews/artifacts/tool-choice-dep-a2-rag-render.png"
rsvg = C.CDLL("librsvg-2.so.2")
cairo = C.CDLL("libcairo.so.2")
gobject = C.CDLL("libgobject-2.0.so.0")

class Rectangle(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("x", "y", "width", "height")]

rsvg.rsvg_handle_new_from_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
rsvg.rsvg_handle_new_from_file.restype = C.c_void_p
rsvg.rsvg_handle_render_document.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Rectangle), C.POINTER(C.c_void_p)]
rsvg.rsvg_handle_render_document.restype = C.c_int
cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
cairo.cairo_image_surface_create.restype = C.c_void_p
cairo.cairo_create.argtypes = [C.c_void_p]
cairo.cairo_create.restype = C.c_void_p
cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
cairo.cairo_surface_write_to_png.restype = C.c_int
cairo.cairo_destroy.argtypes = [C.c_void_p]
cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
cairo.cairo_version_string.restype = C.c_char_p
gobject.g_object_unref.argtypes = [C.c_void_p]
error = C.c_void_p()
handle = rsvg.rsvg_handle_new_from_file(str(source).encode(), C.byref(error))
assert handle and not error.value
surface = cairo.cairo_image_surface_create(0, 800, 330)
context = cairo.cairo_create(surface)
assert rsvg.rsvg_handle_render_document(handle, context, C.byref(Rectangle(0, 0, 800, 330)), C.byref(error)) and not error.value
assert cairo.cairo_surface_write_to_png(surface, str(output).encode()) == 0
cairo.cairo_destroy(context)
cairo.cairo_surface_destroy(surface)
gobject.g_object_unref(handle)
print(json.dumps({"command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-a2-render.py", "source": str(source.relative_to(root)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "render": str(output.relative_to(root)), "render_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "environment": {"renderer": "librsvg-2.so.2 rsvg_handle_render_document", "cairo": cairo.cairo_version_string().decode(), "device": "cpu"}, "result": "800x330 static SVG rendered successfully; CSS animation ignored by renderer"}, indent=2))
