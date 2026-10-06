from pathlib import Path
import ctypes as C
import ctypes.util
import json
import hashlib

base = Path(__file__).resolve().parent
source = base/'inputs/course/figures/p6-19-start-lineage.svg'
rsvg=C.CDLL(ctypes.util.find_library('rsvg-2'))
cairo=C.CDLL(ctypes.util.find_library('cairo'))
gobject=C.CDLL(ctypes.util.find_library('gobject-2.0'))
rsvg.rsvg_handle_new_from_file.argtypes=[C.c_char_p,C.POINTER(C.c_void_p)]
rsvg.rsvg_handle_new_from_file.restype=C.c_void_p
rsvg.rsvg_handle_render_cairo.argtypes=[C.c_void_p,C.c_void_p]
rsvg.rsvg_handle_render_cairo.restype=C.c_int
cairo.cairo_image_surface_create.argtypes=[C.c_int,C.c_int,C.c_int]
cairo.cairo_image_surface_create.restype=C.c_void_p
cairo.cairo_create.argtypes=[C.c_void_p];cairo.cairo_create.restype=C.c_void_p
cairo.cairo_scale.argtypes=[C.c_void_p,C.c_double,C.c_double]
cairo.cairo_surface_write_to_png.argtypes=[C.c_void_p,C.c_char_p];cairo.cairo_surface_write_to_png.restype=C.c_int
cairo.cairo_destroy.argtypes=[C.c_void_p];cairo.cairo_surface_destroy.argtypes=[C.c_void_p]
gobject.g_object_unref.argtypes=[C.c_void_p]
cairo.cairo_version_string.restype=C.c_char_p
error=C.c_void_p()
handle=rsvg.rsvg_handle_new_from_file(str(source).encode(),C.byref(error))
assert handle and not error.value
rows=[]
for width in [640,360]:
    height=round(884*width/640)
    surface=cairo.cairo_image_surface_create(0,width,height)
    cr=cairo.cairo_create(surface)
    cairo.cairo_scale(cr,width/640,width/640)
    assert rsvg.rsvg_handle_render_cairo(handle,cr)
    target=base/f'lineage-{width}.png'
    assert cairo.cairo_surface_write_to_png(surface,str(target).encode())==0
    cairo.cairo_destroy(cr);cairo.cairo_surface_destroy(surface)
    rows.append({'width':width,'height':height,'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
gobject.g_object_unref(handle)
print(json.dumps({'renderer':'system librsvg via Cairo','cairo_version':cairo.cairo_version_string().decode(),'outputs':rows},indent=2))
