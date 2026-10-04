"""External observer for the unchanged product CLI's anonymous urllib requests."""

import atexit
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

events = []


def observe(event, args):
    if event != "urllib.Request":
        return
    url, _, headers, method = args
    names = sorted(headers)
    credential_names = {"authorization", "cookie", "proxy-authorization"}
    if any(name.lower() in credential_names for name in names):
        raise RuntimeError("Anonymous data request unexpectedly contains authentication headers")
    parsed = urlsplit(url)
    events.append(
        {"url_without_query": parsed._replace(query="", fragment="").geturl(), "method": method, "header_names": names, "authentication_headers": []}
    )


sys.addaudithook(observe)


@atexit.register
def save():
    path = os.environ.get("NATURAL_ANONYMOUS_HTTP_RECEIPT")
    if path:
        Path(path).write_text(json.dumps({"scope": "Actual urllib.Request audit events; no opener or response replacement", "requests": events}, indent=2) + "\n")
