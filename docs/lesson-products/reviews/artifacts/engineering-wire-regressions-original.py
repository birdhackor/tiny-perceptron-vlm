"""Current wire fix reruns immutable callback02 finance/lifecycle/paired recovery probes."""
import hashlib
from pathlib import Path
path = Path(__file__).with_name("engineering-runner-probes-callback-02.py")
assert hashlib.sha256(path.read_bytes()).hexdigest() == "d094d72efd49393d85fddc7bb053991685a3e4d178707187dd23c9d8ad84b440"
source = path.read_text().replace("results-callback-02.json", "results-wire-original.json")
exec(compile(source, str(__file__), "exec"), globals())
