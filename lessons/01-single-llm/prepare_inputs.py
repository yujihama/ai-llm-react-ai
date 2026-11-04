#!/usr/bin/env python3
"""教材の資料を連結し、完成入力と検証用のハッシュ一覧を作成する。

実行場所にかかわらず、このスクリプトと同じフォルダー内の learner を
読み、learner/inputs に保存する。既存ファイルと内容が異なる場合は書き込まずに
終了する。資料を変更した後に再構成する場合だけ --overwrite を指定する。
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


DATASET_ID = "LAB1-AONAGI-202604-v1"
BASE_SOURCES = (
    "learner/instructions.txt",
    "learner/initial/P01.txt",
    "learner/initial/E-L01.txt",
    "learner/initial/E-O01.txt",
    "learner/initial/E-O02.txt",
    "learner/initial/E-O03.txt",
    "learner/initial/E-A01.txt",
    "learner/initial/E-R01.txt",
)
SUPPLEMENT_SOURCE = "learner/supplement/E-P01.txt"


class PreparationError(Exception):
    """資料の読み取りや、完成入力の準備を続けられない状態を表す。"""


def read_sources(base_dir):
    """必要な資料をすべて検査し、UTF-8で読み取る。"""
    sources = {}
    errors = []
    for relative_name in BASE_SOURCES + (SUPPLEMENT_SOURCE,):
        path = base_dir / relative_name
        try:
            if not path.is_file():
                errors.append("{}: 資料ファイルがありません。".format(relative_name))
                continue
            sources[relative_name] = path.read_text(encoding="utf-8").strip()
        except UnicodeError:
            errors.append("{}: UTF-8で読み取れません。".format(relative_name))
        except OSError:
            errors.append("{}: 資料を読み取れません。".format(relative_name))
    if errors:
        raise PreparationError("\n".join(errors))
    return sources


def build_outputs(sources):
    """完成入力のバイト列と、それらから計算したマニフェストを返す。"""
    case_b = "\n\n".join(sources[name] for name in BASE_SOURCES) + "\n"
    case_a = case_b.rstrip("\n") + "\n\n" + sources[SUPPLEMENT_SOURCE] + "\n"
    full_input = case_a.encode("utf-8")
    outputs = {
        "case_B.txt": case_b.encode("utf-8"),
        "case_A.txt": full_input,
        "case_B_plus.txt": full_input,
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "inputs": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in sorted(outputs.items())
        },
    }
    outputs["input_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return outputs


def inspect_outputs(output_dir, outputs, overwrite):
    """書き込み前に全出力を比較し、作成・更新するファイルを決める。"""
    if output_dir.exists() and not output_dir.is_dir():
        raise PreparationError("inputs: 保存先がフォルダーではありません。")

    states = {}
    errors = []
    differences = []
    for name, expected in outputs.items():
        path = output_dir / name
        try:
            if not path.exists():
                states[name] = "作成"
            elif not path.is_file():
                errors.append("{}: 同名のファイル以外の項目があります。".format(name))
            elif path.read_bytes() == expected:
                states[name] = "一致"
            else:
                states[name] = "更新"
                differences.append(name)
        except OSError:
            errors.append("{}: 既存ファイルを確認できません。".format(name))

    if differences and not overwrite:
        errors.append(
            "既存内容と異なります: {}。書き込みは行っていません。\n"
            "資料を確認し、意図して再構成する場合は --overwrite を指定してください。"
            .format(", ".join(differences))
        )
    if errors:
        raise PreparationError("\n".join(errors))
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="教材資料から完成入力を準備し、SHA-256とバイト数を記録します。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存内容との差を確認したうえで、資料から意図的に再構成します。",
    )
    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "learner" / "inputs"

    try:
        sources = read_sources(base_dir)
        outputs = build_outputs(sources)
        states = inspect_outputs(output_dir, outputs, args.overwrite)
        # 全資料の読み取りと全出力の比較が終わるまで、書き込みを行わない。
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            if states[name] != "一致":
                try:
                    (output_dir / name).write_bytes(content)
                except OSError:
                    raise PreparationError(
                        "{}: 保存できません。保存済みのファイルがある場合は、"
                        "原因を解消した後に再実行してください。".format(name)
                    ) from None
            print("{}: {}".format(name, states[name]))
        print("case_A.txt / case_B_plus.txt: バイト単位で同一")
    except PreparationError as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1
    except OSError:
        print("エラー: inputs フォルダーを準備できません。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
