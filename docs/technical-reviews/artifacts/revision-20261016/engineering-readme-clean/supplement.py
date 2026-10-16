"""Persist this reviewer's original-source, rendering and dependency checks."""
import ast
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).resolve().parent
WORK = ROOT / "work/engineering-readme-clean"
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_bytes())

def save(name, value):
    (DEST / name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")

def main():
    records = load(WORK / "official/download-receipts.json")
    by_id = {r["id"]:r for r in records}
    result = {"checked_at":datetime.now(UTC).isoformat(),
              "command":"PYTHONPATH=. .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
              "environment":{"python":platform.python_version(),"device":"cpu"},
              "acquisition_method":"Anonymous HTTPS urllib.request.urlopen with inherited proxy and TLS verification; no credentials, writes or remote jobs.",
              "official_downloads":records,
              "prior_document_host_failures":[{"url":"https://docs.astral.sh/uv/guides/projects/","status":403},
                                               {"url":"https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html","status":403}],
              "read_scope":{}}
    # All entire third-party source downloads remain in ignored work/. Only brief supporting excerpts persist.
    snippets=[]
    for name,ranges in [("uv-sync-source",[(1,26),(110,120),(126,145)]),
                        ("torch-tensor-source",[(566,581)])]:
        raw=ROOT / by_id[name]["local"]
        assert sha(raw)==by_id[name]["sha256"]
        lines=raw.read_text().splitlines()
        for start,end in ranges:
            snippets.append({"source_id":name,"url":by_id[name]["url"],"line_start":start,"line_end":end,
                             "excerpt":"\n".join(lines[start-1:end])})
        result["read_scope"][name] = ranges
    save("official-excerpts.json",snippets)

    tree=load(WORK / "official/hf-tree")
    files=[x for x in tree if x["type"]=="file"]
    assert len(files)==16
    names={x["path"].split("/")[-2] for x in files}
    assert names=={"moe-pretrain","moe-sft","moe-joint","dense-joint"}
    expected_names={"model.safetensors","model-config.json","tokenizer.json","inference-manifest.json"}
    for group in names:
        assert {x["path"].split("/")[-1] for x in files if x["path"].split("/")[-2]==group}==expected_names
    result["public_v2"]={"repo":"birdhackor/tiny-perceptron-course-models",
                          "revision":"979cdfacc588ad0536f1c64fff96f264571cf054",
                          "files":files,"export_names":sorted(names),"manifests":{}}
    for group in sorted(names):
        value=load(WORK / ("official/hf-"+group+"-manifest"))
        inspected={k:value[k] for k in ["stage","selected_step","selection","files","selected_checkpoint_sha256"]}
        inspected["origin_kind"]=value["origin"]["kind"]
        assert inspected["origin_kind"]=="all-neural-weights-random"
        assert inspected["selection"]=="validation_loss"
        result["public_v2"]["manifests"][group]=inspected
    for arch in ["moe","dense"]:
        downloaded=load(WORK / ("official/hf-"+arch+"-joint-manifest"))
        original=load(DEST / ("raw/"+arch+"-inference-manifest.json"))
        assert downloaded==original
    # The machine release descriptor is licensing evidence, not a prior review verdict.
    release=next(Path('/workspace/selftrained-v2/outputs/selftrained-v2/operations/release-final-four-exports-attempt-1/monitor/artifact').glob('*/*/receipt.json'))
    data=load(release)
    target=DEST / "raw/four-export-release-receipt.json"
    shutil.copyfile(release,target)
    assert sha(release)==sha(target)
    exports=data["job"]["release"]["exports"]
    licenses=[(e["name"],f["path"],f["license"],f["redistribution_approved"]) for e in exports for f in e["files"]]
    assert len(licenses)==16 and all(v[2]=="MIT" and v[3] is True for v in licenses)
    result["release_license"]={"original":str(release),"copy":str(target.relative_to(ROOT)),"sha256":sha(target),
                               "status":data["status"],"inspected_pointers":["/status","/job/release/exports/*/files/*/{path,license,redistribution_approved}"],
                               "file_licenses":licenses}
    # The CI job record is read fresh from GitHub, then matched with the historical source and retained raw environment fields.
    job=load(WORK / "official/github-macos-job")
    assert job["conclusion"]=="success" and job["status"]=="completed"
    ci=load(ROOT / "docs/validation-artifacts/release-compatibility-ci.json")
    recorded=next(j for j in ci["jobs"] if "macos" in j["job"])
    assert job["id"]==recorded["job_id"]
    historical=WORK / "official/historical-env"
    assert sha(historical)==ci["source_sha256"]["scripts/check_env.py"]
    assert sha(ROOT/'tests/test_env.py')==ci['source_sha256']['tests/test_env.py']
    text=historical.read_text()
    assert "torch.randn(256, 256" in text and ".backward()" in text
    result["macos_ci"]={"job_id":job["id"],"job_name":job["name"],"status":job["status"],
                         "conclusion":job["conclusion"],"head_sha":job["head_sha"],
                         "successful_steps":[s["name"] for s in job["steps"] if s["conclusion"]=="success"],
                         "environment_check":recorded["environment_check"],"historical_check_env_sha256":sha(historical),
                         "scope":"Historical 256x256 matrix forward/backward plus the test_env 32x32 CPU comparison; no new MPS runtime available here."}
    result["read_scope"]["historical-env"]="Full 2906-byte environment-check program; no result commentary."

    # Figure was rendered and opened via view_image in the reviewer session.
    import cairosvg
    from PIL import Image
    figure=ROOT / "course/figures/rewrite-01-character-ids.svg"
    png=WORK / "character-ids.png"
    assert Image.open(png).size==(640,710)
    result["figure"]={"source":str(figure.relative_to(ROOT)),"sha256":sha(figure),"render":str(png.relative_to(ROOT)),
                       "render_sha256":sha(png),"render_size":[640,710],"renderer":"cairosvg "+cairosvg.__version__,
                       "render_command":".venv/bin/python -c \"import cairosvg; cairosvg.svg2png(url='course/figures/rewrite-01-character-ids.svg', write_to='work/engineering-readme-clean/character-ids.png')\"",
                       "viewed_with":"functions.exec tools.view_image; actual displayed image inspected by this reviewer",
                       "observed":"Top then bottom: 貓看狗，狗看貓。 maps to 3,2,1,4,1,2,3,0. Each repeated glyph retains its ID; bidirectional arrows and labels are visible. Static representation says IDs do not order semantic meaning.",
                       "layout_scope":"Actual SVG render only; no claim of a fresh live-site desktop/mobile review. PNG remains ignored and is not submitted."}
    # Short independent reproduction of the figure's convention from lesson 1.1.
    text="貓看狗，狗看貓。"
    vocab=sorted(set(text)); encoded=[vocab.index(c) for c in text]
    assert encoded==[3,2,1,4,1,2,3,0] and ''.join(vocab[i] for i in encoded)==text
    result["figure"]["computed_ids"]=encoded
    # Navigation links in both language versions resolve to files and numbered headings.
    links=[]
    for source in [ROOT/'README.md',ROOT/'README_en.md']:
        for reference in re.findall(r"\]\(([^)]+)\)",source.read_text()):
            if reference.startswith(('http://','https://')):continue
            path,_,fragment=reference.partition('#');p=source.parent/path
            assert p.is_file(),reference
            if fragment and re.fullmatch(r"[A-Z\d]+\.\d+",fragment):
                assert re.search(r'^## '+re.escape(fragment)+r' ',p.read_text(),re.M),reference
            links.append({"source":source.name,"reference":reference})
    result["local_links"]={"count":len(links),"all_resolve":True,"links":links}
    # Parse release configuration and the exact method API nodes, without loading pretrained models.
    release=load(ROOT/'docs/natural-assistant/v4/public-release.json')
    assert release['base_model']['repo']=='Qwen/Qwen3-VL-2B-Instruct'
    assert release['asr_model']['repo']=='openai/whisper-large-v3-turbo'
    assert release['dependency_versions']['torch']=='2.8.0'
    tree=ast.parse((ROOT/'tiny_perceptron/natural_assistant.py').read_bytes())
    methods={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    result['chapter20_route']={'base_model':release['base_model'],'asr_model':release['asr_model'],
                               'dependency_versions':release['dependency_versions'],
                               'inspected_api_nodes':{name:{'lines':[methods[name].lineno,methods[name].end_lineno],
                                'calls':sorted({ast.unparse(n.func) for n in ast.walk(methods[name]) if isinstance(n,ast.Call)})}
                                for name in ['load_core','load_asr']},
                               'scope':'Parse fixed release and inspect actual loader APIs; no model download/training or new Chapter 20 inference.'}
    models=load(ROOT/'docs/course-experiments/public-models.json')['models']
    environments=[]
    for model in models:
        path=ROOT/'docs/course-experiments/results'/ (model['id']+'.json')
        d=load(path)
        environments.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                             **{k:d[k] for k in ['device','torch_version','python_version','gpu','status']}})
    gpu=[e for e in environments if e['device']=='cuda']
    assert gpu and all(e['gpu']=='NVIDIA L4' and e['torch_version']=='2.14.1+cu126' for e in gpu)
    result['historical_experiment_environments']={'records':environments,'cuda_groups':len(gpu),
       'inspected_pointers':['/device','/torch_version','/python_version','/gpu','/status'],
       'scope':'Primitive original environment metadata only; no results or commentary fields read.'}
    # Entrypoint contracts are inspected, not executed as a whole-course gate.
    entries=['build_visuals.py','format_course_code.py','build_course.py','check_course_reviews.py',
             'check_technical_reviews.py','check_notebooks.py']
    assert all((ROOT/'scripts'/p).is_file() for p in entries)
    result['maintenance']={'entrypoints':entries,'whole_course_gate_run':False,
                           'publication_workflow':'.github/workflows/pages.yml',
                           'interpretation':'pages requires reviews before independent kernels/export/site checks. ci has Linux/macOS/Windows core tests. Kernel entrypoint uses one fresh kernel per notebook.'}
    frozen=load(ROOT/'docs/course-revision-20261016/freeze-09.json')
    frozen_page=next(p for p in frozen['pages'] if p['page_id']=='readme')
    assert sha(ROOT/'README.md')==frozen_page['source_sha256']
    result['freeze_match']={'manifest':'docs/course-revision-20261016/freeze-09.json',
                            'source':'README.md','selector':'whole-file','source_sha256':sha(ROOT/'README.md')}
    commands=[[str(ROOT/'.venv/bin/python'),'scripts/check_env.py'],
              ['uv','sync','--frozen','--extra','cpu','--group','notebook','--dry-run']]
    result['setup_commands']=[]
    for command in commands:
        ran=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=40)
        assert ran.returncode==0,(command,ran.stderr)
        result['setup_commands'].append({'argv':command,'returncode':ran.returncode,'stdout':ran.stdout,'stderr':ran.stderr})
    save("supplement-result.json",result)
    print(json.dumps({"status":"passed","public_v2_files":len(files),"licenses":len(licenses),
                      "github_macos":job["conclusion"],"figure":"rendered and viewed","local_links":len(links)}))

if __name__=='__main__':main()
