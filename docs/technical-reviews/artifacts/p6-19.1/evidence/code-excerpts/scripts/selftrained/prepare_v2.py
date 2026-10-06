scripts/selftrained/prepare_v2.py:L32-L79
32: def rebuild(root: Path = ROOT) -> dict:
33:     root = root.resolve()
34:     # Check every fixed program before any producer or filesystem mutation.
35:     for relative in (*BASE_PRODUCERS, *AUGMENTATIONS):
36:         path = root / relative
37:         if path.is_symlink() or not path.is_file():
38:             raise ValueError(f"Missing reviewed reconstruction source: {relative}")
39:     source, output = root / SOURCE_DATA, root / OUTPUT_DATA
40:     for relative in (SOURCE_DATA, OUTPUT_DATA):
41:         path = root
42:         for component in relative.parts:
43:             path /= component
44:             if path.is_symlink():
45:                 raise ValueError("Reconstruction data directories must not be symbolic links")
46:     for relative in BASE_PRODUCERS:
47:         subprocess.run([sys.executable, str(root / relative)], cwd=root, check=True)
48:     if not source.is_dir():
49:         raise ValueError("Original producers did not produce their fixed data directory")
50:     # Fresh copy prevents stale V2 rows/assets from surviving a reconstruction.
51:     for path in source.rglob("*"):
52:         if path.is_symlink():
53:             raise ValueError("Original generated data must not contain symbolic links")
54:     if output.exists():
55:         if not output.is_dir():
56:             raise ValueError("V2 output must be a directory")
57:         shutil.rmtree(output)
58:     output.parent.mkdir(parents=True, exist_ok=True)
59:     shutil.copytree(source, output)
60:     for relative in AUGMENTATIONS:
61:         subprocess.run(
62:             [
63:                 sys.executable,
64:                 str(root / relative),
65:                 "--source-data",
66:                 SOURCE_DATA.as_posix(),
67:                 "--output-data",
68:                 OUTPUT_DATA.as_posix(),
69:             ],
70:             cwd=root,
71:             check=True,
72:         )
73:     return {
74:         "reconstruction_pipeline": "selftrained-v2",
75:         "base_producers": list(BASE_PRODUCERS),
76:         "augmentations": list(AUGMENTATIONS),
77:         "source_data": SOURCE_DATA.as_posix(),
78:         "output_data": OUTPUT_DATA.as_posix(),
79:     }
