"""Render the actual lesson SVG with installed librsvg and Cairo."""

import ctypes
import ctypes.util
import json
from pathlib import Path


class Rectangle(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]


svg = ctypes.CDLL(ctypes.util.find_library("rsvg-2"))
cairo = ctypes.CDLL(ctypes.util.find_library("cairo"))
gobject = ctypes.CDLL(ctypes.util.find_library("gobject-2.0"))
svg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
svg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
svg.rsvg_handle_render_document.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Rectangle), ctypes.POINTER(ctypes.c_void_p)]
svg.rsvg_handle_render_document.restype = ctypes.c_int
cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
cairo.cairo_image_surface_create.restype = ctypes.c_void_p
cairo.cairo_create.argtypes = [ctypes.c_void_p]
cairo.cairo_create.restype = ctypes.c_void_p
cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
cairo.cairo_surface_write_to_png.restype = ctypes.c_int
cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
gobject.g_object_unref.argtypes = [ctypes.c_void_p]
cairo.cairo_version_string.restype = ctypes.c_char_p

root = Path(__file__).resolve().parents[3]
source = root / "course/figures/practical_order.svg"
target = Path(__file__).with_name("natural-11.14-practical-order.png")
error = ctypes.c_void_p()
handle = svg.rsvg_handle_new_from_file(str(source).encode(), ctypes.byref(error))
assert handle and not error.value
surface = cairo.cairo_image_surface_create(0, 1200, 480)
context = cairo.cairo_create(surface)
viewport = Rectangle(0, 0, 1200, 480)
assert svg.rsvg_handle_render_document(handle, context, ctypes.byref(viewport), ctypes.byref(error))
assert not error.value
assert cairo.cairo_surface_write_to_png(surface, str(target).encode()) == 0
cairo.cairo_destroy(context)
cairo.cairo_surface_destroy(surface)
gobject.g_object_unref(handle)
print(json.dumps({"source": str(source), "output": str(target), "size": [1200, 480], "cairo": cairo.cairo_version_string().decode(), "renderer": "system librsvg-2.so.2 rsvg_handle_render_document", "result": "render succeeded"}, indent=2))
