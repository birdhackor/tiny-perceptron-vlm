"""在暫存硬連結副本重建舊發布版本，原樣執行舊審閱契約。

不改舊manifest、hash或判定。覆蓋檔案先unlink，絕不寫入硬連結原檔。
候選manifest可另帶補充還原清單；每筆只能把舊雜湊表記著的位元組換回原位。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "docs/course-publish-20261008"
FREEZE = BASELINE + "/current-source-freeze.json"
OVERLAY = "docs/course-revision-20261016/baseline-overlay.json"
CACHE = "outputs/review-baseline-replay"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def posix_relative(value, label):
    """確認是POSIX相對路徑；每條排除規則各自明寫，不靠正規化前後相等代替。"""
    # 驗證：以/開頭、反斜線、磁碟代號（含C:x）、空段、.段與..段，各報各的規則。
    # 不涵蓋：大小寫別名與檔案系統上的重導向；前者由判重處理，後者由no_redirect處理。
    require(isinstance(value, str), f"{label} 不是字串：{value!r}")
    require(not value.startswith("/"), f"{label} 以 / 開頭：{value!r}")
    require("\\" not in value, f"{label} 含反斜線：{value!r}")
    require(re.match(r"[A-Za-z]:", value) is None, f"{label} 以磁碟代號開頭：{value!r}")
    segments = value.split("/")
    require("" not in segments, f"{label} 含空段：{value!r}")
    require("." not in segments, f"{label} 含 . 段：{value!r}")
    require(".." not in segments, f"{label} 含 .. 段：{value!r}")
    require(PurePosixPath(value).as_posix() == value, f"{label} 不是正規的 POSIX 相對路徑：{value!r}")
    return value


def no_redirect(base, relative, label, scope):
    """從base逐段走到relative；每一段都不能是symlink，實際位置也不能離開base。"""
    # 驗證：每一段的lstat都不是symlink；realpath仍在base底下，擋is_symlink看不到的重導向，例如Windows junction。
    # 不涵蓋：檢查之後才被別的程序換掉的路徑；這裡不防同時修改。
    current = base
    for part in PurePosixPath(relative).parts:
        current = current / part
        require(not current.is_symlink(), f"{label} 經過 symlink：{current.relative_to(base).as_posix()!r}")
    require(
        Path(os.path.realpath(current)).is_relative_to(os.path.realpath(base)),
        f"{label} 實際位置離開 {scope}：{relative!r}",
    )
    return current


def repo_file(root, name, label):
    """讀repo內不經symlink的普通檔案。"""
    source = no_redirect(root, posix_relative(name, label), label, "repo")
    require(source.is_file(), f"{label} 不是普通檔案：{name!r}")
    return source.read_bytes()


def referenced_json(root, reference, label):
    require(isinstance(reference, dict), f"{label} 不是 path／sha256 引用")
    raw = repo_file(root, reference.get("path"), label)
    require(digest(raw) == reference.get("sha256"), f"{label} 的 SHA-256 與引用不符：{reference['path']!r}")
    data = json.loads(raw)
    require(isinstance(data, dict), f"{label} 不是 JSON 物件：{reference['path']!r}")
    return data


def preserved_identities(root):
    """窄用途讀取重放基準記著的兩張舊雜湊表；只驗通到這兩張表的那條引用鏈。"""
    # 驗證：全部從尚未還原的現行root讀；archive_index、catalog、current_technical_inputs三個引用的SHA；
    # freeze01只透過original_path完全相同的唯一一筆index對到repo內路徑，catalog引用、index那筆與讀到的位元組SHA一致。
    # 不涵蓋：review-manifest本身照固定路徑讀，這裡不核它的SHA；index其他項目不驗，留給還原後的composite完整驗證。
    policy = json.loads(repo_file(root, BASELINE + "/review-manifest.json", "重放基準 review-manifest"))
    require(isinstance(policy, dict), "重放基準 review-manifest 不是 JSON 物件")
    index = referenced_json(root, policy.get("archive_index"), "重放基準 archive_index")
    catalog = referenced_json(root, policy.get("catalog"), "重放基準 catalog")
    inputs = referenced_json(root, policy.get("current_technical_inputs"), "重放基準 current_technical_inputs")
    references = catalog.get("manifest_references")
    reference = references.get("freeze01") if isinstance(references, dict) else None
    require(
        isinstance(reference, dict) and isinstance(reference.get("path"), str),
        "重放基準 catalog 缺 manifest_references.freeze01 引用",
    )
    artifacts = index.get("artifacts")
    require(isinstance(artifacts, list), "重放基準 archive_index 的 artifacts 不是 list")
    aliases = [item for item in artifacts if isinstance(item, dict) and item.get("original_path") == reference["path"]]
    require(
        len(aliases) == 1,
        f"重放基準 archive_index 裡 original_path 等於 freeze01 引用的項目有 {len(aliases)} 筆，"
        f"必須恰好一筆：{reference['path']!r}",
    )
    require(
        aliases[0].get("sha256") == reference.get("sha256"),
        f"freeze01 的 catalog 引用與 archive_index 那筆記的 SHA-256 不同：{reference['path']!r}",
    )
    alias = {"path": aliases[0].get("path"), "sha256": reference.get("sha256")}
    freeze01 = referenced_json(root, alias, "重放基準 freeze01")
    tables = {
        "implementation_sha256": freeze01.get("implementation_sha256"),
        "files_sha256": inputs.get("files_sha256"),
    }
    for name, table in tables.items():
        require(
            isinstance(table, dict)
            and table
            and all(isinstance(key, str) and isinstance(value, str) for key, value in table.items()),
            f"舊雜湊表 {name} 不是非空的路徑對 SHA-256 表",
        )
    return tables


def require_preserved_identity(tables, where, entry):
    """補充每筆的(live_path, sha256)必須等於舊雜湊表同一個key的值；兩張表都有這個key時兩邊都要相等。"""
    # 驗證：目的必須是舊表的key，SHA必須是那個key記的值；表上別的key的SHA不算數。
    # 不涵蓋：表上記的位元組本身對不對；補充只能把檔還原成表上的值，不能改內容。
    live_path = entry["live_path"]
    found = [(name, table[live_path]) for name, table in tables.items() if live_path in table]
    require(found, f"{where} 的 live_path 不是舊雜湊表的 key：{live_path!r}")
    for name, expected in found:
        require(entry["sha256"] == expected, f"{where} 的 sha256 與舊雜湊表 {name} 記的值不同：{live_path!r}")


def checked_restores(root, supplements):
    """A段：建立暫存目錄前，對三份清單的每一筆核對路徑形式、來源與重複；補充另核舊身分。"""
    # 驗證：兩份既有清單都經repo_file讀，清單本身與從repo根起的每一段上層都不是symlink、實際位置不離開repo，
    # 而且是普通檔案；不符就在開檔之前拒絕，這時還沒建暫存目錄。
    # 不涵蓋：清單內容與清單本身的SHA，這裡只管讀到的是repo內哪個檔；也不防檢查之後才被換掉的路徑。
    freeze = json.loads(repo_file(root, FREEZE, "還原清單 " + FREEZE))
    # 不另核overlay本身的SHA：要靠新manifest把它列進implementation才綁得住。
    overlay = json.loads(repo_file(root, OVERLAY, "還原清單 " + OVERLAY))
    listed = [
        (f"{FREEZE} {kind}[{name!r}]", entry, False)
        for kind in ("source_files", "criteria", "figures", "notebooks")
        for name, entry in freeze[kind].items()
    ]
    listed += [(f"{OVERLAY} files[{i}]", entry, False) for i, entry in enumerate(overlay["files"])]
    listed += [(f"補充還原清單 files[{i}]", entry, True) for i, entry in enumerate(supplements)]
    tables = preserved_identities(root) if supplements else {}
    restores, owners = [], {}
    for where, entry, supplement in listed:
        require(isinstance(entry, dict), f"{where} 不是物件")
        posix_relative(entry.get("live_path"), where + " 的 live_path")
        # 驗證：來源是repo內不經symlink的普通檔案，位元組等於sha256；之後寫入的就是這份已核對的位元組。
        # 不涵蓋：來源有沒有進Git；沒提交的副本本機會過，乾淨checkout會缺檔。
        raw = repo_file(root, entry.get("path"), where + " 的 path")
        require(digest(raw) == entry.get("sha256"), f"{where} 的來源位元組與 sha256 不符：{entry['path']!r}")
        # 驗證：三份清單的live_path不分大小寫一起比，任何重複都拒絕，同SHA也一樣。
        # 不涵蓋：Unicode正規化不同的同名，以及一筆是另一筆上層目錄的衝突；後者在寫入途中以OSError失敗。
        key = entry["live_path"].casefold()
        if key in owners:
            raise ValueError(f"{where} 的 live_path 與 {owners[key]} 重複（不分大小寫）：{entry['live_path']!r}")
        owners[key] = where
        if supplement:
            require_preserved_identity(tables, where, entry)
        restores.append((where, entry["live_path"], raw))
    return restores


def checked_tracked_names(root):
    """列出要複製進暫存樹的追蹤檔；建立暫存目錄前，先核對每個名稱的上層。"""
    # 驗證：git ls-files每個名稱的上層從repo根起每一段都不是symlink、實際位置不離開repo；
    # 同一個上層只核一次，全部核完才建暫存目錄，之後才有link或copy。
    # 不涵蓋：葉節點本身；它在工作樹裡是symlink時，不論git記成symlink還是普通檔，都照舊複製連結本身、不核它指到哪裡，
    # 暫存樹裡跟著它讀的程式可能讀到repo外。也不防檢查之後才被換掉的上層。
    names = [name for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0") if name]
    checked = set()
    for name in names:
        parent = PurePosixPath(name).parent.as_posix()
        if parent not in checked:
            no_redirect(root, parent, f"追蹤檔 {name!r} 的上層", "repo")
            checked.add(parent)
    return names


def replay(root=ROOT, supplements=()):
    root = Path(root).resolve()
    restores = checked_restores(root, supplements)
    names = checked_tracked_names(root)
    # 驗證：outputs/review-baseline-replay與它在repo內的每一段上層都不是symlink，暫存樹不會建到repo外。
    # 不涵蓋：之後才被換掉的路徑；暫存樹建好後會連同暫存目錄再核一次。
    cache = no_redirect(root, CACHE, "重放暫存根", "repo")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="immutable-baseline-", dir=cache) as folder:
        target = Path(folder)
        for name in names:
            source, destination = root / name, target / name
            if not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        # B段：暫存樹建好之後、任何還原寫入之前。
        # 驗證：暫存目錄從repo根起、每筆live_path的上層從暫存目錄根起，每一段都不是symlink，實際位置也不離開各自的根。
        # 不涵蓋：葉節點本身；它是目錄時unlink會以OSError失敗，是symlink時照舊先unlink再寫，不會跟著它寫出去。
        no_redirect(root, CACHE + "/" + target.name, "重放暫存目錄", "repo")
        for where, live_path, _ in restores:
            parent = PurePosixPath(live_path).parent.as_posix()
            no_redirect(target, parent, where + " 的 live_path 上層", "暫存目錄")
        # 未提交的新sidecar也必須可在新checkout中使用。
        # 硬連結共享inode、沒有寫入隔離：先unlink再寫才不改到原檔，其餘依賴composite只讀。
        for _, live_path, raw in restores:
            destination = target / live_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.write_bytes(raw)
        # 重放執行的是現行root的composite_review與它載入的程式；它們與舊契約不相容時，還原清單補不回來。
        spec = importlib.util.spec_from_file_location(
            "preserved_composite", root / "docs/review-tools/composite_review.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.audit_manifest(target, target / "docs/course-publish-20261008/review-manifest.json", "all")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "failures")}, ensure_ascii=False))
    raise SystemExit(bool(report["failures"]))
