import html
import os
from pathlib import Path
import xml.etree.ElementTree as ET

report = Path("outputs/pytest-results.xml")
lines = ["## Test failure details", ""]
if not report.is_file():
    lines.append("No JUnit report was written. The original failing step and exit code remain authoritative.")
else:
    for case in ET.parse(report).iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        name = case.get("classname", "") + "::" + case.get("name", "")
        lines.extend([
            "<details><summary>" + html.escape(name) + "</summary>",
            "<pre>" + html.escape(failure.text or failure.get("message", "")) + "</pre>",
            "</details>",
            "",
        ])
with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
