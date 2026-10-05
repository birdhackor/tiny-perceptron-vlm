"""Render the frozen original SVG using installed librsvg/Cairo, without a browser."""

import ctypes as C
import ctypes.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
rsvg = C.CDLL(ctypes.util.find_library("rsvg-2"))
cairo = C.CDLL(ctypes.util.find_library("cairo"))
gobject = C.CDLL(ctypes.util.find_library("gobject-2.0"))

class Rectangle(C.Structure):
    _fields_ = [(name, C.c_double) for name in ("x", "y", "width", "height")]

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
cairo.cairo_version_string.restype = C.c_char_p
cairo.cairo_destroy.argtypes = [C.c_void_p]
cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
gobject.g_object_unref.argtypes = [C.c_void_p]

error = C.c_void_p()
handle = rsvg.rsvg_handle_new_from_file(str(HERE / "conversation-updates.svg").encode(), C.byref(error))
assert handle and not error.value
surface = cairo.cairo_image_surface_create(0, 640, 610)
context = cairo.cairo_create(surface)
assert rsvg.rsvg_handle_render_document(handle, context, C.byref(Rectangle(0, 0, 640, 610)), C.byref(error)) == 1
assert not error.value
assert cairo.cairo_surface_write_to_png(surface, str(HERE / "conversation-updates.png").encode()) == 0
cairo.cairo_destroy(context)
cairo.cairo_surface_destroy(surface)
gobject.g_object_unref(handle)
version = ".".join(str(C.c_uint.in_dll(rsvg, name).value) for name in ("rsvg_major_version", "rsvg_minor_version", "rsvg_micro_version"))
print(json.dumps({"python": sys.version, "device": "CPU", "renderer": "installed librsvg/Cairo", "librsvg_version": version, "cairo_version": cairo.cairo_version_string().decode(), "dimensions_pixels": [640, 610], "input": "conversation-updates.svg", "output": "conversation-updates.png"}, indent=2))
