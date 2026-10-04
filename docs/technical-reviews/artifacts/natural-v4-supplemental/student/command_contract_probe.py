"""Small semantic check; no package or repository modifications."""
import importlib.metadata
import json
import subprocess
import sys
from packaging.specifiers import SpecifierSet

result = {
    "environment": {"python":sys.version, "packaging":importlib.metadata.version("packaging"), "device":"cpu"},
    "git_version":subprocess.check_output(["git","--version"],text=True).strip(),
    "PEP440_torch_pins":{"==2.8.0 accepts 2.8.0+cpu":SpecifierSet("==2.8.0").contains("2.8.0+cpu"),"==2.8.0 accepts 2.8.0+cu128":SpecifierSet("==2.8.0").contains("2.8.0+cu128")},
    "scope":"Version-comparison semantics only; no pip install performed"
}
assert all(result["PEP440_torch_pins"].values())
print(json.dumps(result,indent=2))
