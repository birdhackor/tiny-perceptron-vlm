"""以直接索引核對切塊、排序、還原與課文算例。"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import patchify, scene, unpatchify


def reference_patches(images, p):
    """依座標逐值取出；不用 reshape 或 permute 建立參考答案。"""
    b, c, h, w = images.shape
    return torch.tensor(
        [
            [
                [
                    images[batch, channel, row * p + inner_row, col * p + inner_col].item()
                    for channel in range(c)
                    for inner_row in range(p)
                    for inner_col in range(p)
                ]
                for row in range(h // p)
                for col in range(w // p)
            ]
            for batch in range(b)
        ],
        dtype=images.dtype,
    )


def main():
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/10.2.md"
    body = frozen.read_bytes()
    fences = body.decode().split("```python\n")[1:]
    excerpt = fences[1].split("```", 1)[0]
    definitions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(excerpt).body
        if isinstance(node, ast.FunctionDef)
    }
    definition_matches = {}
    for function in (patchify, unpatchify):
        definition_matches[function.__name__] = definitions[function.__name__] == ast.dump(
            ast.parse(inspect.getsource(function)).body[0], include_attributes=False
        )
    assert all(definition_matches.values()), "課文函式與匯入實作不同"

    image = scene("red", "square")[None]
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    expected_red = ((yy >= 4) & (yy <= 12) & (xx >= 4) & (xx <= 12)).to(image.dtype)
    assert torch.equal(image[0, 0], expected_red), "紅色區域與生成規則不同"
    assert image[0, 1:].count_nonzero().item() == 0, "非紅色通道應為零"
    assert torch.equal(image, scene("red", "square").unsqueeze(0)), "新增筆數軸的結果不同"
    scene_result = {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "red_pixels": int(image[0, 0].sum().item()),
        "none_equals_unsqueeze": True,
    }

    exercises = []
    for p in (4, 8):
        patches = patchify(image, p)
        restored = unpatchify(patches, 3, 16, 16, p)
        expected_shape = [1, (16 // p) ** 2, 3 * p * p]
        assert list(patches.shape) == expected_shape, "切塊形狀不同"
        assert torch.equal(patches, reference_patches(image, p)), "切塊座標或通道順序不同"
        assert torch.equal(image, restored), "還原數值不同"
        exercises.append({"patch_size": p, "shape": expected_shape, "roundtrip_equal": True})

    distinct = torch.arange(3 * 16 * 16).reshape(1, 3, 16, 16)
    split = distinct.reshape(1, 3, 4, 4, 4, 4)
    reordered = split.permute(0, 2, 4, 1, 3, 5)
    flattened = reordered.reshape(1, -1, 48)
    inverse_split = flattened.reshape(1, 4, 4, 3, 4, 4)
    inverse_reordered = inverse_split.permute(0, 3, 1, 4, 2, 5)
    assert torch.equal(flattened, reference_patches(distinct, 4)), "展開結果與座標參考不同"
    assert torch.equal(inverse_reordered.reshape(1, 3, 16, 16), distinct), "逆向軸順序不同"
    assert distinct.data_ptr() == split.data_ptr(), "本例第一次 reshape 預期共用儲存"
    stages = {
        "input_shape": list(distinct.shape),
        "split_shape": list(split.shape),
        "split_shares_storage": distinct.data_ptr() == split.data_ptr(),
        "permuted_shape": list(reordered.shape),
        "flattened_shape": list(flattened.shape),
        "inverse_permuted_shape": list(inverse_reordered.shape),
        "first_patch": flattened[0, 0].tolist(),
        "direct_reshape_first_position": distinct.reshape(1, 16, 48)[0, 0].tolist(),
        "direct_reshape_coordinates": "channel=0, rows=0..2, columns=0..15",
    }

    matrix = torch.tensor([[1, 2], [3, 4]])
    transposition = {
        "reshape": matrix.reshape(4).tolist(),
        "permute": matrix.permute(1, 0).tolist(),
        "permute_then_reshape": matrix.permute(1, 0).reshape(4).tolist(),
        "permute_equals_T": torch.equal(matrix.permute(1, 0), matrix.T),
    }
    assert transposition["reshape"] == [1, 2, 3, 4], "攤平次序不同"
    assert transposition["permute_then_reshape"] == [1, 3, 2, 4], "交換軸後攤平次序不同"

    grid = torch.arange(16.0).reshape(1, 1, 4, 4)
    row_order = patchify(grid, 2)
    column_order = row_order[:, [0, 2, 1, 3]]
    compatible_restored = unpatchify(column_order[:, [0, 2, 1, 3]], 1, 4, 4, 2)
    assert torch.equal(grid, compatible_restored), "相容的依欄切拼應可還原"
    assert row_order[0, 1].tolist() == [2, 3, 6, 7], "依列排序的右上塊不符"
    assert column_order[0, 1].tolist() == [8, 9, 12, 13], "依欄排序的左下塊不符"
    ordering = {
        "grid": grid[0, 0].tolist(),
        "row_order": row_order[0].tolist(),
        "column_order": column_order[0].tolist(),
        "compatible_column_roundtrip_equal": True,
    }

    general_cases = []
    for b, c, h, w, p in ((1, 1, 4, 4, 2), (2, 3, 6, 10, 2), (1, 3, 16, 16, 4), (1, 3, 16, 16, 8)):
        values = torch.arange(b * c * h * w, dtype=torch.int64).reshape(b, c, h, w)
        for transpose in (False, True):
            x = values.transpose(-1, -2) if transpose else values
            actual = patchify(x, p)
            assert torch.equal(actual, reference_patches(x, p)), "逐座標參考結果不符"
            assert torch.equal(unpatchify(actual, c, x.shape[-2], x.shape[-1], p), x), "逐值往返結果不符"
            general_cases.append({"shape": list(x.shape), "patch_size": p, "contiguous": x.is_contiguous(), "exact_equal": True})

    invalid_geometry = []
    for h, w in ((15, 16), (16, 15), (15, 15)):
        try:
            patchify(torch.zeros(1, 3, h, w), 4)
        except ValueError as error:
            invalid_geometry.append({"shape": [1, 3, h, w], "error": str(error)})
        else:
            raise AssertionError("高寬不能整除時應報錯")

    unused = {}
    exec("b, _, _ = (1, 16, 48)", unused)
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-2/verify_patch_order.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "implementation_sha256": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest(),
        "excerpt_definitions_equal": definition_matches,
        "scene": scene_result,
        "exercises": exercises,
        "stages": stages,
        "transposition": transposition,
        "ordering": ordering,
        "general_cases": general_cases,
        "invalid_geometry": invalid_geometry,
        "unused_name_binding": {"b": unused["b"], "_": unused["_"]},
        "result": "所有逐值、形狀、排序與錯誤條件核對通過",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
