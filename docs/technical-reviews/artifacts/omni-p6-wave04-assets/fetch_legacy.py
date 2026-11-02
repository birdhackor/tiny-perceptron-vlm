"""Read each existing chapter manifest's exact public namespace, metadata only."""

import json
import posixpath
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_primary import CACHE, ROOT, acquire


def main():
    models=json.loads((ROOT/'docs/course-experiments/public-models.json').read_bytes())['models']
    requests=[]
    for model in models:
        prefix=posixpath.commonpath([posixpath.dirname(item['path']) for item in model['files']])
        url=f'https://huggingface.co/api/models/{model["repo"]}/tree/{model["revision"]}/{prefix}?recursive=true&expand=false'
        requests.append((f'hf-course-{model["id"]}-tree.json',url))
    with ThreadPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(acquire,requests))
    (Path(__file__).parent/'legacy-primary-receipts-v1.json').write_text(json.dumps({'scope':'One small subtree GET per pinned chapter model; no payload download.','sources':results},ensure_ascii=False,indent=2)+'\n')
    for row in results:
        print(row['name'],row.get('bytes'),row.get('error','OK'))
    assert all('error' not in row for row in results)


if __name__=='__main__':
    main()
