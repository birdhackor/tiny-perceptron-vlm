"""Validate Modal 1.6 option registration locally, with remote entry blocked."""

import inspect
import json
import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

sys.path.insert(0, "/tmp/omni-infra-review-initial-20261029")
import modal
from scripts.omni_revision import preflight


class ReviewRemoteBlocked(Exception):
    pass


@contextmanager
def no_remote(*args, **kwargs):
    raise ReviewRemoteBlocked("Independent review never enters the Modal runtime")
    yield


output = {"modal_version": modal.__version__}
for name, method in [("App.function", modal.App.function), ("App.run", modal.App.run),
                     ("Volume.from_name", modal.Volume.from_name),
                     ("Volume.objects.create", modal.Volume.objects.create),
                     ("Volume.objects.delete", modal.Volume.objects.delete),
                     ("Volume.objects.list", modal.Volume.objects.list),
                     ("Workspace.billing.summary", modal.Workspace.from_context().billing.summary)]:
    output[name] = str(inspect.signature(method))
facade = types.SimpleNamespace(App=modal.App, Image=modal.Image, Secret=modal.Secret,
    Volume=types.SimpleNamespace(from_name=modal.Volume.from_name,
        objects=types.SimpleNamespace(list=lambda **kw: [], delete=lambda *a, **kw: None)))
with patch.object(modal.App, "run", no_remote):
    output["local_registration_result"] = preflight.run_cpu_probe(
        facade, "independent-sdk-registration-only", "review/dummy", "dummy-never-hydrated")
assert output["local_registration_result"]["controller_error_type"] == "ReviewRemoteBlocked"
print(json.dumps(output, indent=2))
