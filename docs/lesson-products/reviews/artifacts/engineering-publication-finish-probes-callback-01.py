"""Informed rerun of exact original six terminal/cancel/race probes against current publisher."""
import hashlib
from pathlib import Path
original = Path(__file__).with_name("engineering-publication-finish-probes-original.py")
assert hashlib.sha256(original.read_bytes()).hexdigest() == "d25d67fb9a6dc4929fae56a6a1f55d7a7ae23e0ea274016011f76de8bde4ff7a"
source = original.read_text()
assert source.count("engineering-publication-finish-results-original.json") == 1
source = source.replace("engineering-publication-finish-results-original.json", "engineering-publication-finish-results-callback-01.json")
exec(compile(source, str(__file__), "exec"), globals())
