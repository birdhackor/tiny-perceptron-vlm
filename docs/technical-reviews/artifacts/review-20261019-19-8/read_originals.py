"""以 curl 讀取原始來源；只保存少量摘錄與取得收據，不保存第三方全文。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.blocks = []
        self.current = None
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if self.current is None and tag in ('p', 'h1', 'h2', 'h3', 'h4'):
            self.current = [tag, [], dict(attrs).get('id', '')]
        self.stack.append(tag)

    def handle_data(self, data):
        if self.current is not None and not self.skip:
            self.current[1].append(data)

    def handle_endtag(self, tag):
        if self.current is not None and tag == self.current[0]:
            value = re.sub(r'\s+', ' ', ''.join(self.current[1])).strip()
            if value:
                self.blocks.append((self.current[0], value, self.current[2]))
            self.current = None
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1


SOURCES = [
    ('mtp', 'https://arxiv.org/html/2404.19737v1',
     ['In practice, our architecture', 'To study this phenomenon', 'To allow fair comparisons', 'Considering multi-token prediction']),
    ('dpo', 'https://arxiv.org/html/2305.18290v3',
     ['DPO pipeline', 'SFT', 'reference model', 'classification loss']),
    ('rlhf', 'https://arxiv.org/html/2203.02155v1',
     ['supervised fine-tuning', 'PPO', 'alignment tax', 'pretraining']),
    ('ppo', 'https://arxiv.org/html/1707.06347v2',
     ['surrogate objective', 'sample', 'policy']),
    ('sdpa', 'https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html',
     ['fused', 'numerical accuracy', 'implementation', 'dropout_p']),
]


def main():
    receipts = []
    for name, url, needles in SOURCES:
        command = ['curl', '-fLsS', '--max-time', '45', url]
        result = subprocess.run(command, capture_output=True)
        receipt = {'id': name, 'url': url, 'command': command,
                   'accessed_on': datetime.now(timezone.utc).date().isoformat(),
                   'returncode': result.returncode}
        if result.returncode:
            receipt['error'] = result.stderr.decode(errors='replace')
            print(name, receipt['error'])
        else:
            raw = result.stdout
            receipt['original_sha256'] = hashlib.sha256(raw).hexdigest()
            receipt['original_bytes'] = len(raw)
            parsed = Blocks()
            parsed.feed(raw.decode())
            matches = []
            for tag, text, identifier in parsed.blocks:
                if tag == 'p' and any(n.casefold() in text.casefold() for n in needles):
                    matches.append(text)
            print('\nSOURCE', name, url)
            for passage in matches[:8]:
                print(passage[:2600])
            receipt['excerpt'] = ' '.join(matches[0].split()[:24]) if matches else ''
            receipt['matched_paragraphs'] = len(matches)
        receipts.append(receipt)
    destination = Path(__file__).with_name('original-source-receipts.json')
    destination.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
