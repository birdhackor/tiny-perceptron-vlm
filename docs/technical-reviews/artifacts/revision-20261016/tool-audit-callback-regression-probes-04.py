"""重用immutable03的gate/Notebook反例，於真實current reader06有效基底回歸。

只調整測試支援圖的當前收件seal，不改原探針、來源、程式或舊報告。
未重跑reader.py未變的CLI/figure/helper區段。
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def main():
    audit = ROOT / "docs/course-revision-20261016/reviews/tool-audit-callback-03.json"
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == "90c2dc8bc790b95953c8026e2d713fd68a84bbd9b64ba9a06f604563e7bcd719"
    prior = json.loads(audit.read_bytes())
    source_ref = next(x for x in prior["artifacts"] if x["path"].endswith("tool-audit-callback-probes-03.py"))
    source_path = ROOT / source_ref["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_ref["sha256"]
    source = source_path.read_text()
    prefix = source.split("        supplement = subprocess.run(", 1)[0]
    notebook = source.split("        # 只用硬連結建立暫存副本；", 1)[1].split("        reader = load_module(", 1)[0]
    trailer = "    output = {" + source.split("    output = {", 1)[1]
    generated = prefix + "        # 只用硬連結建立暫存副本；" + notebook + trailer
    generated = generated.replace("manifest-reader-05.json", "manifest-reader-06.json")
    generated = generated.replace("tool-audit-callback-probes-results-03.json", "tool-audit-callback-regression-probes-results-04.json")
    generated = generated.replace("tool-audit-callback-probes-03.py", "tool-audit-callback-regression-probes-04.py")
    generated = generated.replace("本階段確實尚未有 technical/continuity seals，不應通過 all。", "真正reader-only候選尚無全部technical/all證據，不應通過all；未假稱不存在已提供後期seal。")
    old = '''            for wave in m["review_waves"]:
                seal = read(wave["seals"]["reader"])
                seal["report_sha256s"] = [new["sha256"] if x == old["sha256"] else x for x in seal["report_sha256s"]]
                wave["seals"]["reader"] = save(seal)
'''
    new = '''            for wave in m["review_waves"]:
                for seal_role in list(wave["seals"]):
                    seal = read(wave["seals"][seal_role])
                    seal["report_sha256s"] = [new["sha256"] if x == old["sha256"] else x for x in seal["report_sha256s"]]
                    wave["seals"][seal_role] = save(seal)
'''
    assert generated.count(old) == 1
    generated = generated.replace(old, new)
    old_closure = '''            proof["checkpoint_closure"]["report"] = save(closure)
'''
    new_closure = '''            old_ref = proof["checkpoint_closure"]["report"].copy()
            new_ref = save(closure)
            proof["checkpoint_closure"]["report"] = new_ref
            for wave in m["review_waves"]:
                for seal_role in list(wave["seals"]):
                    seal = read(wave["seals"][seal_role])
                    if old_ref["sha256"] in seal["report_sha256s"]:
                        seal["report_sha256s"].append(new_ref["sha256"])
                    wave["seals"][seal_role] = save(seal)
'''
    assert generated.count(old_closure) == 2
    generated = generated.replace(old_closure, new_closure)
    closure_control = '''        def closure_graph_positive(m):
            proof = page(m)["reader"]
            closure = read(proof["checkpoint_closure"]["report"])
            closure["independent_fixture_marker"] = "nonsemantic metadata changes hash without changing judgment"
''' + new_closure + '''            m["issue_reports"].append(new_ref)

        run("rebased_closure_graph_positive_control", closure_graph_positive, "pass", "只對單頁closure追加非語義fixture metadata並更換path/SHA，seal保留其他頁共用的舊hash且追加新hash；支援圖本身須350頁pass。")

'''
    marker = "        def closure_issue_omission(m):"
    assert generated.count(marker) == 1
    generated = generated.replace(marker, closure_control + marker)
    target = ART / "tool-audit-callback-regression-probes-results-04.json"
    assert not target.exists()
    namespace = {"__file__": str(__file__), "__name__": "independent_generated_regressions"}
    exec(compile(generated, str(__file__) + "::derived03", "exec"), namespace)
    namespace["main"]()
    output = json.loads(target.read_bytes())
    output["derivation"] = {"immutable_probe_source": source_ref,
                            "generated_source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                            "retained_sections": ["gate mutations through closure-before-observation", "Notebook clone/control mutation"],
                            "support_changes": ["use true current manifest-reader-06", "rebind every provided role seal for path-only supporting graph changes", "append new closure hash in every applicable provided seal while preserving the old group hash and omitting new issue-inventory membership", "add hash-changing nonsemantic single-page closure graph positive control"],
                            "skipped": ["unchanged reader CLI", "reader figure display", "helper plan/context/completion", "phase7 missing-pointer route"]}
    output["unexpected_results"] = [r["id"] for r in output["results"]
                                    if (r["expected_contract_result"] == "pass" and (r["failures"] or r.get("record_count") != 350))
                                    or (r["expected_contract_result"] == "reject" and not r["failures"])]
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"results": len(output["results"]), "unexpected_results": output["unexpected_results"],
                      "reader_controls": [{"id": r["id"], "status": r["actual_status"], "records": r.get("record_count"), "failures": r["failures"]}
                                          for r in output["results"] if r["expected_contract_result"] == "pass"],
                      "result_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False))
    return 0 if not output["unexpected_results"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
