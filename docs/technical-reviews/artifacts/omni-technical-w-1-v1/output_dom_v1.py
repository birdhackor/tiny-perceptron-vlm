"""Compare output content/order, preserving exact text and image attributes.

Whitespace between output blocks is renderer formatting. Whitespace within a
label or pre block remains significant. Unexpected visible text is rejected.
"""
from html.parser import HTMLParser


class OutputDOM(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.output_depth = None
        self.current = None
        self.capture = None
        self.parts = []
        self.outputs = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            self.depth += 1
            if "output" in attrs.get("class", "").split():
                assert self.current is None, "nested output"
                self.current = []
                self.output_depth = self.depth
        if self.current is None:
            return
        if tag == "pre" or (tag == "p" and "output-label" in attrs.get("class", "").split()):
            assert self.capture is None, "nested captured block"
            self.capture = tag
            self.parts = []
        elif tag == "img":
            assert self.capture is None
            self.current.append(["image", attrs.get("src"), attrs.get("alt")])

    def handle_endtag(self, tag):
        if self.current is not None and tag == self.capture:
            self.current.append(["pre" if tag == "pre" else "label", "".join(self.parts)])
            self.capture = None
            self.parts = []
        if tag == "div":
            if self.depth == self.output_depth:
                assert self.capture is None
                self.outputs.append(self.current)
                self.current = None
                self.output_depth = None
            self.depth -= 1

    def handle_data(self, text):
        if self.capture:
            self.parts.append(text)
        elif self.current is not None:
            assert not text.strip(), "unclassified visible output text"


def outputs(body):
    parser = OutputDOM()
    parser.feed(body)
    parser.close()
    assert parser.current is None and parser.capture is None
    return parser.outputs
