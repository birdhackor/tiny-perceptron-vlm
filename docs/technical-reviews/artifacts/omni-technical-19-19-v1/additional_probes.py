"""19.19 的小型補充核對；不啟動訓練。"""

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.inference import Product

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-19-19-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    result = {'created_at': datetime.now(UTC).isoformat(),
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu'}}
    products = {}
    for name in ('b', 'c'):
        value = Product(ROOT / 'outputs/lesson-products' / name, name)
        state = torch.load(value.directory/'model.pt', weights_only=True, map_location='cpu')
        before = {key: tensor.clone() for key, tensor in value.model.state_dict().items()}
        output = value.solve('兩盒追蹤;A=2;B=6;事件=交換;查詢=B', 3) if name == 'b' else value.judge('卡片進度;申請編號=?', 'info')
        products[name] = {
            'eval_mode': not value.model.training,
            'state_keys_equal_parameter_keys': set(state) == dict(value.model.named_parameters()).keys(),
            'no_registered_buffers': not dict(value.model.named_buffers()),
            'parameters_unchanged_by_inference': all(torch.equal(tensor, value.model.state_dict()[key]) for key,tensor in before.items()),
            'output': output,
        }
    result['inference_only'] = products
    torch.manual_seed(312)
    dropout = torch.nn.Dropout(0.5).eval()
    x = torch.ones(20)
    samples = torch.multinomial(torch.tensor([0.5,0.5]),20,replacement=True)
    result['eval_not_sampling_control'] = {'eval_dropout_is_identity':torch.equal(dropout(x),x),'twenty_samples':samples.tolist(),'number_of_distinct_samples':len(set(samples.tolist())), 'scope':'只示範 eval 不控制顯式 multinomial；不宣稱 B/C 採抽樣（實際使用 argmax/排序）。'}
    copy = CACHE/'tamper-b'
    shutil.copytree(ROOT/'outputs/lesson-products/b',copy)
    path = copy/'auto-policy.json'
    path.write_bytes(path.read_bytes()+b' ')
    try:
        Product(copy,'b')
    except ValueError as error:
        result['manifest_tamper_rejected'] = {'mutated':'auto-policy.json appended space','error':str(error)}
    else:
        raise AssertionError('tamper not rejected')
    historical = ROOT/'docs/course-revision-20261029-omni/evidence/legacy-bc-public-examples.json'
    old = json.loads(historical.read_bytes())
    result['historical_stdout_integrity'] = []
    for entry in old['executions']:
        record = {'case':entry['case']}
        for stream in ('stdout','stderr'):
            receipt = entry[stream]
            path = Path(receipt['path'])
            record[stream] = {'path':str(path),'exists':path.exists(),'expected_sha256':receipt['sha256']}
            if path.exists():
                record[stream].update(actual_sha256=sha(path),sha256_matches=sha(path)==receipt['sha256'])
        result['historical_stdout_integrity'].append(record)
    freeze_path = ROOT/'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
    freeze = json.loads(freeze_path.read_bytes())
    result['source_pins'] = {'freeze':{'path':str(freeze_path.relative_to(ROOT)),'sha256':sha(freeze_path)},'pages':[]}
    for page_id,selector in [('19.19','19.19'),('19.11','19.11'),('19.24','19.24'),('first-steps','W.1')]:
        page = next(p for p in freeze['pages'] if p['page_id']==page_id)
        raw = (ROOT/page['snapshot']).read_text()
        pattern = rf'(?ms)^## {re_escape(selector)} .*?(?=^## |\Z)'
        import re
        snapshot_slice = re.search(pattern,raw).group()
        current_slice = re.search(pattern,(ROOT/page['source']).read_text()).group()
        assert snapshot_slice == current_slice
        result['source_pins']['pages'].append({'page_id':page_id,'source':page['source'], 'snapshot':page['snapshot'], 'snapshot_sha256':sha(ROOT/page['snapshot']), 'selector':'section:'+selector,'slice_sha256':hashlib.sha256(snapshot_slice.encode()).hexdigest(),'current_slice_matches':True,'inspected_scope': '19.19 全文；19.11 推論／續訓角色與 CPU 操作文字；19.24 最後產品保存與待實測範圍；W.1 CPU 安裝及 activate 指令。前置圖未作本節證據。'})
    result['installed_official_source'] = []
    for function in (torch.load,torch.nn.Module.eval,torch.nn.Module.train):
        path = Path(inspect.getsourcefile(function))
        lines,line = inspect.getsourcelines(function)
        result['installed_official_source'].append({'api':function.__qualname__,'path':str(path),'sha256':sha(path),'start_line':line,'end_line':line+len(lines)-1,'runtime_version':torch.__version__,'declared_git':torch.version.git_version,'note':'實裝原始碼；宣告 commit 的 raw GitHub docs 路徑 404，另讀官方 v2.9.0 固定版支持共同契約，未冒稱外網核過此 commit。'})
    for command in [
        [sys.executable,'scripts/selftrained/chat.py','--help'],
        [sys.executable,'-m','scripts.lesson_products.train_local','--help'],
        [sys.executable,'-m','scripts.lesson_products.train_local','--job','docs/lesson-products/jobs/train-b-v1.json','--output','outputs/lesson-products/b','--device','cpu'],
    ]:
        completed = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=30)
        result.setdefault('safe_cli_probes',[]).append({'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'scope':'help 或既有輸出目錄拒絕；未進入資料生成或訓練。'})
    assert result['safe_cli_probes'][0]['exit_code'] == 0
    assert result['safe_cli_probes'][1]['exit_code'] == 0
    assert result['safe_cli_probes'][2]['exit_code'] != 0
    assert '輸出需另立新目錄' in result['safe_cli_probes'][2]['stderr']
    assert all(record['eval_mode'] and record['state_keys_equal_parameter_keys'] and record['parameters_unchanged_by_inference'] for record in products.values())
    output = ART/'additional-probes-v1.json'
    with output.open('x') as file:
        json.dump(result,file,ensure_ascii=False,indent=2)
        file.write('\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'status':'passed','historical_stdout_matches':[x['stdout'].get('sha256_matches') for x in result['historical_stdout_integrity']]}))


def re_escape(value):
    import re
    return re.escape(value)


if __name__ == '__main__':
    main()
