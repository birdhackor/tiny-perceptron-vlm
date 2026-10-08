"""只查 Secret 名稱與 HF_TOKEN 是否存在，不讀取或輸出 token 內容。"""

import os
from pathlib import Path

import modal


def main():
    environment = os.environ.get("MODAL_ENVIRONMENT", "main")
    preferred = os.environ.get("HF_MODAL_SECRET") or "tiny-perceptron-hf"
    names = [secret.name for secret in modal.Secret.objects.list(environment_name=environment)]
    candidates = [preferred] if preferred in names else names
    matches = []
    for name in candidates:
        try:
            modal.Secret.from_name(name, environment_name=environment, required_keys=["HF_TOKEN"]).hydrate()
            matches.append(name)
        except (modal.exception.InvalidError, modal.exception.NotFoundError):
            continue
    if len(matches) != 1:
        raise RuntimeError(
            f"Modal {environment} 環境中找到 {len(matches)} 個含 HF_TOKEN 的 Secret；"
            "請建立 tiny-perceptron-hf，或以 Repository variable HF_MODAL_SECRET 指定名稱。"
        )
    selected = matches[0]
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
        stream.write(f"HF_MODAL_SECRET={selected}\n")
    print(f"使用 Modal Secret：{selected}（已確認 HF_TOKEN 存在，未讀取內容）")


if __name__ == "__main__":
    main()
