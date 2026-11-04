#!/usr/bin/env python3
"""基本検索の取得断片を親節へ広げ、必要に応じて明示参照を辿る。

固定文字chunkの索引と順位はbasic_retrieve.pyをそのまま使用する。
parentはseedと重なる節、linkedはそれに加えて同一文書内の明示参照を取得する。
対応表は原文のハッシュ・文字位置・参照元行を検証してから使用する。
検索や参照関係から回答の正しさを判定するプログラムではない。
Python 3標準ライブラリだけで動作し、外部APIやLLMを呼び出さない。
"""

import argparse
from collections import deque
import json
from pathlib import Path
import re
import sys

from basic_retrieve import (
    RetrievalError, json_bytes, load_index, normalize, original_lines,
    read_utf8, retrieve, within,
)


OUTPUT_NAMES = ("index.json", "retrieval.json", "retrieved.txt", "prompt.txt", "context.json")


def _require(condition, message):
    if not condition:
        raise RetrievalError("structure: " + message)


def load_structure(path, corpus, index):
    """全対応表を検証する。未選択の文書や節の不正も見逃さない。"""
    raw, digest = read_utf8(path, "structure")
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        raise RetrievalError("structure: 正しいJSONではありません。") from None
    _require(isinstance(value, dict), "最上位はオブジェクトにしてください。")
    _require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
             "schema_versionは1にしてください。")
    _require(isinstance(value.get("documents"), list), "documentsは配列にしてください。")
    known = {document["source"]: document for document in index["documents"]}
    mapped = {}
    texts = {}
    for document in value["documents"]:
        _require(isinstance(document, dict), "documentsの各要素はオブジェクトにしてください。")
        source = document.get("source")
        _require(isinstance(source, str) and source in known, "sourceにcorpus内の相対資料パスを指定してください。")
        _require(source not in mapped, "同じsourceを重複して定義できません: " + source)
        expected_hash = document.get("sha256")
        _require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is not None,
                 "sha256が不正です: " + source)
        text, actual_hash = read_utf8(corpus / source, source)
        _require(expected_hash.lower() == actual_hash == known[source]["sha256"],
                 "原文のsha256が一致しません。対応表を原文と照合してください: " + source)
        sections = document.get("sections")
        _require(isinstance(sections, list), "sectionsは配列にしてください: " + source)
        validated = []
        identifiers = set()
        for section in sections:
            _require(isinstance(section, dict), "sectionはオブジェクトにしてください: " + source)
            section_id = section.get("id")
            _require(isinstance(section_id, str) and bool(section_id.strip()), "節のidが空です: " + source)
            _require(section_id not in identifiers, "節のidが重複しています: " + source + "/" + section_id)
            identifiers.add(section_id)
            _require(isinstance(section.get("title"), str), "節のtitleは文字列にしてください: " + source)
            start, end = section.get("start_offset"), section.get("end_offset")
            _require(type(start) is int and type(end) is int and 0 <= start < end <= len(text),
                     "節の文字範囲が不正です: " + source + "/" + section_id)
            references = section.get("references")
            _require(isinstance(references, list), "referencesは配列にしてください: " + source + "/" + section_id)
            clean_refs = []
            for reference in references:
                _require(isinstance(reference, dict), "referenceはオブジェクトにしてください: " + source)
                target, line = reference.get("target"), reference.get("evidence_line")
                _require(isinstance(target, str) and bool(target.strip()), "参照先targetが不正です: " + source)
                _require(isinstance(line, str) and re.fullmatch(r"L\d+", line) is not None,
                         "evidence_lineはL番号で指定してください: " + source)
                clean_refs.append({"target": target, "evidence_line": line})
            validated.append({"id": section_id, "title": section["title"],
                              "start_offset": start, "end_offset": end, "references": clean_refs})
        validated.sort(key=lambda section: section["start_offset"])
        cursor = 0
        lines = original_lines(text)
        for section in validated:
            _require(section["start_offset"] == cursor,
                     "節は原文全体を隙間・重複なく覆う必要があります: " + source)
            cursor = section["end_offset"]
            for reference in section["references"]:
                _require(reference["target"] in identifiers,
                         "同一文書内に参照先の節がありません: " + source + "/" + reference["target"])
                matching = [line for line in lines if line["label"] == reference["evidence_line"]]
                _require(len(matching) == 1 and section["start_offset"] <= matching[0]["start_offset"]
                         and matching[0]["end_offset"] <= section["end_offset"],
                         "参照元L番号は参照元の節に完全に含まれる一意の行である必要があります: "
                         + source + "/" + section["id"] + "/" + reference["evidence_line"])
        _require(cursor == len(text), "節が原文の末尾まで覆っていません: " + source)
        mapped[source] = {"sections": validated, "by_id": {s["id"]: s for s in validated}}
        texts[source] = text
    return mapped, texts, digest


def expand_context(selected, mapped, texts, mode, max_sections, structure_hash):
    """seedの親節を先に選び、その後に選択済み節の明示参照をFIFOで辿る。"""
    units = []
    by_key = {}
    queue = deque()
    log = []
    duplicates = []
    omitted = []
    unexpanded_refs = []
    warnings = []
    warned_sources = set()
    seeds = []

    def add(candidate, reason):
        key = (candidate["source"], candidate["kind"], candidate["section_id"],
               candidate["start_offset"], candidate["end_offset"])
        event = {"source": candidate["source"], "section_id": candidate["section_id"],
                 "start_offset": candidate["start_offset"], "end_offset": candidate["end_offset"],
                 "reason": reason}
        if key in by_key:
            unit = by_key[key]
            unit["reasons"].append(reason)
            event.update({"action": "deduplicated", "unit_id": unit["unit_id"]})
            duplicates.append(dict(event))
        elif len(units) >= max_sections:
            event.update({"action": "omitted", "cause": "max_sections"})
            omitted.append(dict(event))
            unit = None
        else:
            unit = dict(candidate)
            unit.update({"unit_id": "unit-{:06d}".format(len(units) + 1), "reasons": [reason], "seed_chunk_ids": []})
            units.append(unit)
            by_key[key] = unit
            if unit["kind"] == "section":
                queue.append(unit)
            event.update({"action": "selected", "unit_id": unit["unit_id"]})
        log.append(event)
        return unit

    def section_candidate(source, section):
        start, end = section["start_offset"], section["end_offset"]
        return {"kind": "section", "source": source, "section_id": section["id"],
                "title": section["title"], "start_offset": start, "end_offset": end,
                "text": texts[source][start:end]}

    for rank, (score, chunk) in enumerate(selected, start=1):
        seeds.append({"seed_rank": rank, "chunk_id": chunk["chunk_id"], "source": chunk["source"],
                      "start_offset": chunk["start_offset"], "end_offset": chunk["end_offset"], "score": score})
        source = chunk["source"]
        reason = {"type": "seed_overlap", "seed_rank": rank, "seed_chunk_id": chunk["chunk_id"]}
        if source not in mapped:
            if source not in warned_sources:
                warnings.append({"code": "document_not_mapped", "source": source,
                                 "message": "対応表にない資料のため、取得chunkをそのまま候補にします。"})
                warned_sources.add(source)
            candidate = {"kind": "raw_chunk", "source": source, "section_id": None, "title": None,
                         "start_offset": chunk["start_offset"], "end_offset": chunk["end_offset"], "text": chunk["text"]}
            add(candidate, {**reason, "type": "unmapped_seed_chunk"})
        else:
            for section in mapped[source]["sections"]:
                if section["start_offset"] < chunk["end_offset"] and chunk["start_offset"] < section["end_offset"]:
                    add(section_candidate(source, section), dict(reason))

    # queueには重複排除後の選択済み節だけが入る。循環しても同じ節を再処理しない。
    while queue:
        unit = queue.popleft()
        section = mapped[unit["source"]]["by_id"][unit["section_id"]]
        for reference in section["references"]:
            relation = {"source": unit["source"], "from_unit_id": unit["unit_id"],
                        "from_section_id": unit["section_id"], "target_section_id": reference["target"],
                        "evidence_line": reference["evidence_line"]}
            if mode == "parent":
                unexpanded_refs.append({**relation, "cause": "parent_mode"})
                continue
            reason = {"type": "explicit_reference", **relation}
            target = mapped[unit["source"]]["by_id"][reference["target"]]
            acquired = add(section_candidate(unit["source"], target), reason)
            if acquired is None:
                unexpanded_refs.append({**relation, "cause": "max_sections"})

    # 取得経路を保存したうえで、循環を含む経路にもseedとの対応を付ける。
    # スコアを参照先へ付け替えることはしない。
    origins = {unit["unit_id"]: {reason["seed_chunk_id"] for reason in unit["reasons"]
                                if "seed_chunk_id" in reason} for unit in units}
    changed = True
    while changed:
        changed = False
        for unit in units:
            current = origins[unit["unit_id"]]
            size = len(current)
            for reason in unit["reasons"]:
                if reason["type"] == "explicit_reference":
                    current.update(origins[reason["from_unit_id"]])
            changed = changed or len(current) != size
    for unit in units:
        unit["seed_chunk_ids"] = [seed["chunk_id"] for seed in seeds if seed["chunk_id"] in origins[unit["unit_id"]]]
    for seed in seeds:
        seed["selected_unit_ids"] = [unit["unit_id"] for unit in units if seed["chunk_id"] in unit["seed_chunk_ids"]]

    return {
        "schema_version": 1, "mode": mode, "structure_sha256": structure_hash,
        "budget": {"max_sections": max_sections, "selected_units": len(units),
                   "unit": "重複排除後の節または対応表にないraw chunk。両方を1件と数える。",
                   "omitted_attempts": len(omitted), "limit_reached": len(units) == max_sections},
        "selection_order": "全seedを順位順・節開始位置順に処理した後、明示参照を記載順のFIFOで処理する。",
        "presentation_order": "本文の原文断片はsource・start_offset・end_offsetの昇順に表示する。unit_idと選択履歴は維持する。",
        "score_scope": "検索スコアはseed chunkだけの値。節または参照先のスコアではない。",
        "offset_unit": "原文のUnicodeコードポイント。開始を含み、終了を含まない。",
        "deduplication_key": "source・kind・section_id・start_offset・end_offsetが同一の取得単位。",
        "seeds": seeds, "selected_units": units, "expansion_log": log,
        "deduplicated": duplicates, "omitted_units": omitted,
        "unexpanded_references": unexpanded_refs, "warnings": warnings,
    }


def context_text(context):
    """選択した原文だけを掲載し、検索seedと取得理由を区別して示す。"""
    parts = [
        "検索方式: 文字bigramのTF-IDFとcosine（字面由来の疎ベクトル）\n",
        "文脈取得方式: {}\n".format(context["mode"]),
        "取得単位数: {} / 上限 {}\n".format(len(context["selected_units"]), context["budget"]["max_sections"]),
        "文字位置は開始を含み終了を含まない。検索スコアはseed chunkだけの値であり、節や参照先の評価ではない。\n",
    ]
    if not context["seeds"]:
        parts.append("\n一致するbigramを持つ検索結果はありません。資料の不存在を示すものではありません。\n")
    for seed in context["seeds"]:
        parts.append("seed {}: {} / {} [{}, {}) / 検索スコア {:.12g}\n".format(
            seed["seed_rank"], seed["chunk_id"], seed["source"], seed["start_offset"], seed["end_offset"], seed["score"]))
    for warning in context["warnings"]:
        parts.append("注意: {}: {}\n".format(warning["source"], warning["message"]))
    if context["omitted_units"]:
        parts.append("注意: 取得上限により追加できなかった候補が{}件あります。参照が揃ったことを保証しません。\n".format(len(context["omitted_units"])))
    for unit in sorted(context["selected_units"], key=lambda item: (item["source"], item["start_offset"], item["end_offset"])):
        parts.extend(["\n[取得 {}]\n".format(unit["unit_id"]), "資料: {}\n".format(unit["source"]),
                      "節: {}\n".format(unit["section_id"] if unit["kind"] == "section" else "対応表なし・原文chunk"),
                      "原文文字位置: [{}, {})\n".format(unit["start_offset"], unit["end_offset"])])
        for reason in unit["reasons"]:
            if reason["type"] == "explicit_reference":
                parts.append("取得理由: {}の節{}、{}からの明示参照\n".format(
                    reason["source"], reason["from_section_id"], reason["evidence_line"]))
            else:
                parts.append("取得理由: seed {}（{}）{}\n".format(
                    reason["seed_rank"], reason["seed_chunk_id"],
                    "と重なる親節" if reason["type"] == "seed_overlap" else "を原文のまま保持"))
        parts.extend(["--- 原文断片 ---\n", unit["text"], "\n--- 原文断片終わり ---\n"])
    return "".join(parts)


def save_outputs(out, corpus, protected, outputs, overwrite):
    """5ファイルすべてを先に確認し、差分があれば標準では一切書き込まない。"""
    if out.exists() and not out.is_dir():
        raise RetrievalError("out: 保存先がフォルダーではありません。")
    states = {}
    differences = []
    for name in OUTPUT_NAMES:
        path = out / name
        resolved = path.resolve()
        if within(resolved, corpus) or resolved in protected or path.is_symlink():
            raise RetrievalError(name + ": 入力資料またはリンク先への書き込みはできません。")
        if path.exists() and any(path.samefile(source) for source in protected):
            raise RetrievalError(name + ": 入力資料と同じファイルへの書き込みはできません。")
        if not path.exists():
            states[name] = "作成"
        elif not path.is_file():
            raise RetrievalError(name + ": 同名のファイル以外の項目があります。")
        elif path.read_bytes() == outputs[name]:
            states[name] = "一致"
        else:
            states[name] = "更新"
            differences.append(name)
    if differences and not overwrite:
        raise RetrievalError("既存内容と異なります: {}。書き込みは行っていません。\n"
                             "別の --out を使うか、意図して再構成する場合は --overwrite を指定してください。"
                             .format(", ".join(differences)))
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if states[name] != "一致":
            try:
                (out / name).write_bytes(outputs[name])
            except OSError:
                raise RetrievalError(name + ": 保存できません。保存済みファイルがある場合は原因を解消して再実行してください。") from None
        print("{}: {}".format(name, states[name]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--structure", required=True, type=Path, help="原文のハッシュ・節の位置・明示参照を記したJSON")
    parser.add_argument("--mode", required=True, choices=("parent", "linked"))
    parser.add_argument("--max-sections", type=int, default=20, help="raw chunkを含む取得単位数の上限（既定: 20）")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or args.top_k < 1 or args.max_sections < 1:
        parser.error("--chunk-size、--top-k、--max-sections は1以上で指定してください。")
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap は0以上、--chunk-size 未満で指定してください。")
    if not normalize(args.query):
        parser.error("--query は空白以外の文字を含めてください。")
    try:
        corpus, base, structure, out = (p.resolve() for p in (args.corpus, args.base, args.structure, args.out))
        if within(out, corpus):
            raise RetrievalError("out: corpusと同じ場所や、その配下には保存できません。")
        base_text, base_hash = read_utf8(base, "base")
        if not base_text.strip():
            raise RetrievalError("base: 入力本文が空です。")
        index = load_index(corpus, args.chunk_size, args.overlap)
        mapped, texts, structure_hash = load_structure(structure, corpus, index)
        selected, report = retrieve(index, args.query, args.top_k)
        report["base_sha256"] = base_hash
        context = expand_context(selected, mapped, texts, args.mode, args.max_sections, structure_hash)
        context["base_sha256"] = base_hash
        text = context_text(context)
        outputs = {"index.json": json_bytes(index), "retrieval.json": json_bytes(report),
                   "retrieved.txt": text.encode("utf-8"),
                   "prompt.txt": (base_text + "\n\n【検索結果】\n" + text).encode("utf-8"),
                   "context.json": json_bytes(context)}
        protected = {base, structure} | {(corpus / doc["source"]).resolve() for doc in index["documents"]}
        save_outputs(out, corpus, protected, outputs, args.overwrite)
        print("seed {} 件 → 取得 {} 単位（LLMの呼び出しなし）".format(len(selected), len(context["selected_units"])))
        return 0
    except RetrievalError as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1
    except OSError:
        print("エラー: 資料または保存先にアクセスできません。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
