from pathlib import Path
import ctypes, re
from PIL import Image
base = Path(__file__).parent
rsvg = ctypes.CDLL("librsvg-2.so.2")
cairo = ctypes.CDLL("libcairo.so.2")
gobj = ctypes.CDLL("libgobject-2.0.so.0")
rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
rsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
rsvg.rsvg_handle_render_cairo.restype = ctypes.c_int
cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
cairo.cairo_image_surface_create.restype = ctypes.c_void_p
cairo.cairo_create.argtypes = [ctypes.c_void_p]
cairo.cairo_create.restype = ctypes.c_void_p
cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
gobj.g_object_unref.argtypes = [ctypes.c_void_p]
images = []
for name in ("rag", "tools", "architecture_policy_update"):
    source = Path("course/figures") / (name + ".svg")
    w, h = map(int, re.search(r'viewBox="0 0 ([0-9]+) ([0-9]+)"', source.read_text()).groups())
    handle = rsvg.rsvg_handle_new_from_file(str(source).encode(), None)
    assert handle
    surface = cairo.cairo_image_surface_create(0, w, h)
    context = cairo.cairo_create(surface)
    assert rsvg.rsvg_handle_render_cairo(handle, context)
    out = base / (name + ".png")
    assert cairo.cairo_surface_write_to_png(surface, str(out).encode()) == 0
    cairo.cairo_destroy(context); cairo.cairo_surface_destroy(surface); gobj.g_object_unref(handle)
    images.append(Image.open(out).convert("RGB"))
    print(name, w, h)
combined = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), "white")
y = 0
for im in images:
    combined.paste(im, (0, y)); y += im.height
combined.save(base / "prerequisite-figures.png")
for name in ("rag", "tools", "architecture_policy_update"):
    (base / (name + ".png")).unlink()
