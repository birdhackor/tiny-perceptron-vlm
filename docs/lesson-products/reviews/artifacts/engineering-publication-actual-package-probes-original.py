"""真正式包的唯讀正例：不載權重、不連HF/Modal、不重跑評估。"""
import hashlib
import json
import platform
import sys
import tempfile
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
TREE=Path('/workspace/work/lesson-products-transport')
ART=Path(__file__).parent
sys.path.insert(0,str(TREE))
from scripts.lesson_products.publication import sha,unpack_verified,verify_release


def main():
 start=datetime.now(UTC).isoformat()
 spec_path=TREE/'docs/lesson-products/jobs/release-v1.json';spec=json.loads(spec_path.read_bytes())
 package_path=TREE/'docs/lesson-products/results/publication-package-v1.json';package=json.loads(package_path.read_bytes())
 archive=ROOT/'outputs/lesson-products/lesson-products-v1-20261016-release.tar.gz'
 directory=ROOT/'outputs/lesson-products/public-v1-release'
 assert sha(archive)==spec['transport_sha256']==package['archive_sha256']
 assert archive.stat().st_size==spec['asset_bytes']==package['archive_bytes']==20761964
 direct=verify_release(directory,spec)
 assert direct['files']==32 and direct['gates']==package['verification']['gates']
 actual={p.relative_to(directory).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(directory.rglob('*')) if p.is_file()}
 assert actual==spec['files']==package['files']
 assert sum(item['bytes'] for item in actual.values())==package['unpacked_bytes']==21799410
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='actual-publication-package-audit-') as temporary:
  extracted=unpack_verified(archive,spec['transport_sha256'],Path(temporary)/'extracted')
  verified=verify_release(extracted,spec)
  assert verified==direct
  extracted_files={p.relative_to(extracted).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(extracted.rglob('*')) if p.is_file()}
  assert extracted_files==actual
  # Source-bound negative sanity check only in our temp copy, preserving all actual outputs.
  tamper=extracted/'b/model.pt';tamper.write_bytes(tamper.read_bytes()+b'local audit tamper')
  error=None
  try:verify_release(extracted,spec)
  except ValueError as raised:error=str(raised)
  assert error
 report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed actual-input engineering check','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'source_worktree':str(TREE),'checked_files':[{'path':str(p),'sha256':sha(p)} for p in [TREE/'scripts/lesson_products/publish.py',TREE/'scripts/lesson_products/publication.py',spec_path,package_path,archive]],'local_public_directory':str(directory),'actual_archive_bytes':archive.stat().st_size,'files':actual,'actual_package_verification':verified,'temporary_tamper_rejected':True,'temporary_tamper_error':error,'all_expected':True,'limits':['No model/Torch load, training/evaluation/GPU/paid/Modal/HF or external write; only actual package local extraction/hash/contract checks.','Formal gates are report values checked for required acceptance, not independently recomputed scientific metrics in this engineering task.','Remote asset bytes not personally downloaded; separate API metadata reads and Actions roundtrip receipt are explicitly distinguished.']}
 path=ART/'engineering-publication-actual-package-results-original.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'files':32,'archive_sha256':spec['transport_sha256'],'actual_package_passed':True,'gates':verified['gates'],'receipt_sha256':sha(path)}))


if __name__=='__main__':main()
