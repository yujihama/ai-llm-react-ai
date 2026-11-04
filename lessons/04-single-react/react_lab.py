#!/usr/bin/env python3
"""単一LLMの行動JSONを人が中継する、オフラインのReAct実習用ランタイム。

モデルやAPIを呼び出さず、ローカルの検索・原文取得・値比較だけを実行する。
行動のpurposeは確認目的の短文であり、内部思考の全文を要求しない。
next_prompt.txtは固定指示・手続・初期証跡・全行動観測履歴から再構成する。
"""

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from basic_retrieve import RetrievalError, load_index, original_lines, retrieve

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUTPUTS = ("state.json", "next_prompt.txt", "trace.jsonl", "latest_observation.txt")
STATUSES = {"条件充足", "条件不充足", "判定保留"}
COLLECTIONS = {"policy", "evidence"}


class LabError(Exception):
    """状態や入力資料の問題により、安全に処理を続けられない。"""


class ActionError(Exception):
    """次の行動で修正可能なschemaまたはtoolのエラー。"""


def encoded(value, pretty=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2 if pretty else None, separators=None if pretty else (",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inside(path, parent):
    return path == parent or parent in path.parents


def learner_path(value):
    if not isinstance(value, str) or not value:
        raise LabError("設定の資料パスが不正です。")
    path = (ROOT / value).resolve()
    if not inside(path, ROOT / "learner") or any(part.lower() in {"teacher", "instructor"} for part in path.parts):
        raise LabError("入力資料はこの教材のlearner配下に限定します。")
    return path


def session_path(value):
    path = (ROOT / value).resolve()
    if path == ROOT / "runs" or not inside(path, ROOT / "runs"):
        raise LabError("session/outはこの教材のruns配下の個別フォルダーにしてください。")
    return path


def relative(path):
    return path.relative_to(ROOT).as_posix()


def utf8(path):
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        raise LabError("資料をUTF-8で読み取れません: " + path.name) from None


def strict_json(raw):
    def invalid_constant(value):
        raise ValueError("JSONには非有限数を指定できません。")
    value = json.loads(raw, parse_constant=invalid_constant)
    encoded(value)  # 浮動小数点のオーバーフローも、保存できないJSONとして拒否する。
    return value


def load_json(path):
    try:
        return strict_json(utf8(path))
    except (ValueError, RecursionError):
        raise LabError("JSONの構文または数値表現が不正です: " + path.name) from None


def ranges_union(spans):
    result = []
    for start, end in sorted((s["start_offset"], s["end_offset"]) for s in spans):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def complete_lines(raw, spans):
    """固定L番号の本文全体が、観測済み範囲の和に含まれる行だけを返す。"""
    coverage = ranges_union(spans)
    labels = {}
    for line in original_lines(raw):
        if line["label"]:
            number = int(line["label"][1:])
            if number in labels:
                raise LabError("原文内の固定L番号が重複しています。")
            end = line["end_offset"]
            while end > line["start_offset"] and raw[end - 1] in "\r\n":
                end -= 1
            labels[number] = any(start <= line["start_offset"] and end <= stop for start, stop in coverage)
    return sorted(number for number, visible in labels.items() if visible)


def document_text(state, document):
    info = state["catalog"][document]
    raw = utf8(learner_path(info["source"]))
    if digest(raw.encode("utf-8")) != info["sha256"]:
        raise LabError("原文ハッシュが変化しました。新しいsessionを開始してください。")
    return raw


def observe(state, document, source, raw, start, end):
    info = state["catalog"][document]
    record = state["observed"].setdefault(document, {
        "document": document, "source": info["source"], "sha256": info["sha256"], "spans": []})
    span = {"start_offset": start, "end_offset": end, "text": raw[start:end]}
    if span not in record["spans"]:
        record["spans"].append(span)
    return {"document": document, "source": source, "sha256": info["sha256"], **span,
            "complete_lines_in_this_excerpt": complete_lines(raw, [span]),
            "complete_lines_observed_so_far": complete_lines(raw, record["spans"])}


def validate_structure(value, texts, catalog):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1 or not isinstance(value.get("documents"), list):
        raise LabError("構造マップのschemaが不正です。")
    mapped = {}
    for document in value["documents"]:
        if not isinstance(document, dict) or not isinstance(document.get("source"), str):
            raise LabError("構造マップのsourceが不正です。")
        name = Path(document["source"]).name
        if name not in catalog or name in mapped or document.get("sha256") != catalog[name]["sha256"]:
            raise LabError("構造マップの原文・ハッシュ・重複を確認してください。")
        sections = document.get("sections")
        if not isinstance(sections, list):
            raise LabError("構造マップのsectionsが不正です。")
        clean = {}
        for section in sections:
            if not isinstance(section, dict) or not isinstance(section.get("id"), str) or not section["id"] or section["id"] in clean:
                raise LabError("構造マップの節IDが不正です。")
            start, end = section.get("start_offset"), section.get("end_offset")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(texts[name]):
                raise LabError("構造マップの原文範囲が不正です。")
            clean[section["id"]] = {"start_offset": start, "end_offset": end}
        cursor = 0
        for section in sorted(clean.values(), key=lambda s: s["start_offset"]):
            if section["start_offset"] != cursor:
                raise LabError("構造マップの節は全原文を重複・隙間なく覆う必要があります。")
            cursor = section["end_offset"]
        if cursor != len(texts[name]):
            raise LabError("構造マップが原文末尾まで覆っていません。")
        mapped[name] = clean
    return mapped


def initial_state(scenario_path):
    scenario = load_json(scenario_path)
    required = {"id", "initial_documents", "instructions", "procedure", "collections", "structure", "max_steps"}
    if not isinstance(scenario, dict) or not required <= scenario.keys():
        raise LabError("scenarioの必須項目が不足しています。")
    if not isinstance(scenario["id"], str) or not scenario["id"] or type(scenario["max_steps"]) is not int or scenario["max_steps"] < 1:
        raise LabError("scenarioのid/max_stepsが不正です。")
    if not isinstance(scenario["initial_documents"], list) or not scenario["initial_documents"]:
        raise LabError("initial_documentsには初期証跡を指定してください。")
    if not isinstance(scenario["collections"], dict) or set(scenario["collections"]) != COLLECTIONS:
        raise LabError("collectionsはpolicyとevidenceを指定してください。")
    manifest = {relative(scenario_path): digest(scenario_path.read_bytes())}

    def read_input(path):
        raw = utf8(path)
        manifest[relative(path)] = digest(raw.encode("utf-8"))
        return raw

    catalog, texts = {}, {}

    def register(path):
        raw = read_input(path)
        name = path.name
        if path.suffix.lower() != ".txt":
            raise LabError("証跡と検索資料は.txtにしてください。")
        info = {"source": relative(path), "sha256": digest(raw.encode("utf-8")), "characters": len(raw)}
        if name in catalog and catalog[name]["sha256"] != info["sha256"]:
            raise LabError("同じファイル名が異なる原文を指しています: " + name)
        catalog.setdefault(name, info)
        texts[name] = raw
        complete_lines(raw, [])
        return name, info, raw

    initial = []
    initial_ids = set()
    for value in scenario["initial_documents"]:
        path = learner_path(value)
        name, info, raw = register(path)
        if name in initial_ids:
            raise LabError("初期証跡のファイル名は一意にしてください。")
        initial_ids.add(name)
        initial.append({"document": name, **info, "text": raw})
    collections, collection_paths = {}, {}
    for collection, value in scenario["collections"].items():
        folder = learner_path(value)
        if not folder.is_dir() or not inside(folder, ROOT / "learner/corpus"):
            raise LabError("collectionはlearner/corpus配下のフォルダーにしてください。")
        collection_paths[collection] = relative(folder)
        entries = {}
        for path in sorted(folder.rglob("*.txt")):
            resolved = learner_path(relative(path))
            if not inside(resolved, folder):
                raise LabError("collectionの外部リンクは利用できません。")
            name, info, raw = register(resolved)
            if name in entries:
                raise LabError("collection内のファイル名は一意にしてください。")
            entries[name] = info
        collections[collection] = entries
    instructions_path = learner_path(scenario["instructions"])
    procedure_path = learner_path(scenario["procedure"])
    instructions = read_input(instructions_path)
    procedure = read_input(procedure_path)
    structure_path = learner_path(scenario["structure"])
    structure_raw = read_input(structure_path)
    try:
        structure = strict_json(structure_raw)
    except (ValueError, RecursionError):
        raise LabError("構造マップのJSONが不正です。") from None
    state = {"schema_version": 1, "scenario_id": scenario["id"], "status": "active", "max_steps": scenario["max_steps"],
             "step_count": 0, "instructions": instructions, "procedure": procedure, "initial_documents": initial,
             "catalog": catalog, "collections": collections, "collection_paths": collection_paths,
             "structure": validate_structure(structure, texts, catalog), "source_manifest": manifest,
             "observed": {}, "history": [], "fingerprints": [], "final_result": None}
    for document in initial:
        observe(state, document["document"], document["document"], document["text"], 0, len(document["text"]))
    return state


def next_prompt(state):
    parts = [state["instructions"].rstrip(),
             "【実行状態】\nケースC\n単一LLMが次の行動を選び、人がJSONを中継します。ランタイムはモデルを呼び出しません。\n"
             "実行済み試行数: {} / {}。残り: {}。状態: {}。".format(state["step_count"], state["max_steps"], max(0, state["max_steps"] - state["step_count"]), state["status"]),
             "【評価手続】\n" + state["procedure"].rstrip(), "【初期証跡】"]
    for doc in state["initial_documents"]:
        parts.append("資料: {}\nsha256: {}\n{}".format(doc["document"], doc["sha256"], doc["text"]))
    parts.append("【行動と観測の全履歴】\n" + ("まだ行動はありません。" if not state["history"] else ""))
    for entry in state["history"]:
        parts.append(encoded(entry, pretty=True).decode("utf-8"))
    if state["status"] == "active":
        parts.append("【次の出力】\n残り手数内で、次の行動JSONを1個だけ出力してください。")
    elif state["status"] == "finished":
        parts.append("【終了】\n条件別評価を受け付けました。内容の正誤や規程適用は人がレビューします。追加行動は受け付けません。")
    else:
        parts.append("【未完了】\n手数上限までにfinishが受理されませんでした。評価結果を自動生成しません。追加行動は受け付けません。")
    return "\n\n".join(parts) + "\n"


def trace_bytes(state):
    return b"".join(encoded(entry) for entry in state["history"])


def latest_observation(state):
    observation = state["history"][-1]["observation"] if state["history"] else {"status": "not_started", "note": "まだツール観測はありません。"}
    return encoded({"observation": observation, "remaining_steps": max(0, state["max_steps"] - state["step_count"]),
                    "session_status": state["status"]}, pretty=True)


def save_session(folder, state, create=False):
    state.pop("state_sha256", None)
    state["state_sha256"] = digest(encoded(state))
    outputs = {"state.json": encoded(state, pretty=True), "next_prompt.txt": next_prompt(state).encode("utf-8"),
               "trace.jsonl": trace_bytes(state), "latest_observation.txt": latest_observation(state)}
    if create and folder.exists():
        raise LabError("既存のsessionは初期化できません。新しいruns配下の名前を指定してください。")
    protected = [learner_path(path) for path in state["source_manifest"]]
    for name in OUTPUTS:
        path = folder / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise LabError("出力先にリンクまたはファイル以外の項目があります。")
        if path.exists() and (path.stat().st_nlink > 1 or any(path.samefile(source) for source in protected)):
            raise LabError("共有されたファイルや入力資料を出力で上書きできません。")
        temporary = folder / (name + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise LabError("未完了の保存ファイルがあります。sessionを確認してください。")
    folder.mkdir(parents=True, exist_ok=not create)
    # stateを最後に確定する。途中の障害は次回の整合性検査で検出し、成功扱いしない。
    for name in OUTPUTS:
        with (folder / (name + ".tmp")).open("xb") as handle:
            handle.write(outputs[name])
    for name in ("next_prompt.txt", "trace.jsonl", "latest_observation.txt", "state.json"):
        os.replace(folder / (name + ".tmp"), folder / name)


def load_session(folder):
    for name in OUTPUTS:
        path = folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink > 1:
            raise LabError("sessionの必須ファイルがないか、リンクになっています。")
    state = load_json(folder / "state.json")
    if not isinstance(state, dict):
        raise LabError("stateの形式が不正です。")
    signature = state.pop("state_sha256", None)
    if signature != digest(encoded(state)):
        raise LabError("stateのハッシュが一致しません。履歴は変更していません。")
    if state.get("schema_version") != 1 or state.get("step_count") != len(state.get("history", [])):
        raise LabError("stateのschemaまたは履歴件数が不正です。")
    if ((folder / "next_prompt.txt").read_bytes() != next_prompt(state).encode("utf-8")
            or (folder / "trace.jsonl").read_bytes() != trace_bytes(state)
            or (folder / "latest_observation.txt").read_bytes() != latest_observation(state)):
        raise LabError("stateとprompt/traceが一致しません。履歴は変更していません。")
    for source, expected in state["source_manifest"].items():
        path = learner_path(source)
        if digest(path.read_bytes()) != expected:
            raise LabError("入力資料のハッシュが変化しました。新しいsessionを開始してください。")
    for collection, folder_name in state["collection_paths"].items():
        folder_path = learner_path(folder_name)
        actual = set()
        for path in folder_path.rglob("*.txt"):
            resolved = learner_path(relative(path))
            if not inside(resolved, folder_path):
                raise LabError("collectionの外部リンクは利用できません。")
            actual.add(relative(resolved))
        if actual != {info["source"] for info in state["collections"][collection].values()}:
            raise LabError("collectionの資料構成が変化しました。新しいsessionを開始してください。")
    for document, record in state["observed"].items():
        raw = document_text(state, document)
        if record["sha256"] != state["catalog"][document]["sha256"] or any(
                not (0 <= span["start_offset"] <= span["end_offset"] <= len(raw)) or
                span["text"] != raw[span["start_offset"]:span["end_offset"]] for span in record["spans"]):
            raise LabError("観測済み原文spanの整合性が失われています。")
    return state


def require(condition, message):
    if not condition:
        raise ActionError(message)


def validate_refs_shape(refs):
    require(isinstance(refs, list), "refsは配列にしてください。")
    for ref in refs:
        require(isinstance(ref, dict) and set(ref) == {"document", "lines"}, "refsにはdocumentとlinesを指定してください。")
        require(isinstance(ref["document"], str) and Path(ref["document"]).name == ref["document"] and ref["document"].endswith(".txt"), "documentは.txt付きファイル名だけを指定してください。")
        require(isinstance(ref["lines"], list) and ref["lines"] and all(type(n) is int and n >= 1 for n in ref["lines"]), "linesは1以上の固定L番号の配列にしてください。")


def validate_action(action):
    require(isinstance(action, dict), "行動はJSONオブジェクト1個にしてください。")
    require(isinstance(action.get("purpose"), str) and 0 < len(action["purpose"].strip()) <= 200 and len(action["purpose"]) <= 200, "purposeは確認目的の短文を200文字以内で指定してください。")
    kind = action.get("type")
    fields = {"search": {"collection", "query", "top_k"}, "read": {"collection", "document"},
              "compare": {"op", "value_type", "left", "right", "refs"}, "finish": {"conditions", "summary"}}
    require(isinstance(kind, str) and kind in fields, "typeはsearch/read/compare/finishから選んでください。")
    required = fields[kind] | {"type", "purpose"}
    require(required <= action.keys() and set(action) <= required | ({"sections"} if kind == "read" else set()), "行動の必須項目または項目名を確認してください。")
    if kind in {"search", "read"}:
        require(isinstance(action["collection"], str) and action["collection"] in COLLECTIONS, "collectionはpolicyまたはevidenceです。")
    if kind == "search":
        require(isinstance(action["query"], str) and bool(action["query"].strip()), "queryは空白以外の文字を含めてください。")
        require(type(action["top_k"]) is int and 1 <= action["top_k"] <= 3, "top_kは1〜3の整数です。")
    elif kind == "read":
        document = action["document"]
        require(isinstance(document, str) and Path(document).name == document and document.endswith(".txt") and "/" not in document and "\\" not in document, "documentには.txt付きのファイル名だけを指定してください。")
        if "sections" in action:
            require(isinstance(action["sections"], list) and action["sections"] and all(type(n) is int and n >= 1 for n in action["sections"]) and len(set(action["sections"])) == len(action["sections"]), "sectionsは重複のない1以上の条番号配列です。省略時は全文です。")
    elif kind == "compare":
        require(action["op"] in ("lt", "le", "eq") and action["value_type"] in ("number", "datetime", "text"), "op/value_typeの値を確認してください。")
        require(action["value_type"] != "text" or action["op"] == "eq", "textはeqだけを使用できます。")
        validate_refs_shape(action["refs"])
        require(bool(action["refs"]), "比較の根拠refsが必要です。")
    else:
        require(isinstance(action["summary"], str) and bool(action["summary"].strip()), "summaryを指定してください。")
        conditions = action["conditions"]
        require(isinstance(conditions, list) and len(conditions) == 4, "conditionsはC1〜C4の4件です。")
        ids = []
        for condition in conditions:
            require(isinstance(condition, dict) and set(condition) == {"id", "status", "explanation", "refs", "missing"}, "各条件にはid/status/explanation/refs/missingを指定してください。")
            require(isinstance(condition["id"], str) and condition["id"] in {"C1", "C2", "C3", "C4"}, "条件IDはC1〜C4です。")
            ids.append(condition["id"])
            require(isinstance(condition["status"], str) and condition["status"] in STATUSES, "条件のstatusを確認してください。")
            require(isinstance(condition["explanation"], str) and bool(condition["explanation"].strip()), "explanationを指定してください。")
            validate_refs_shape(condition["refs"])
            require(isinstance(condition["missing"], list) and all(isinstance(x, str) and x.strip() for x in condition["missing"]), "missingは不足情報・追加確認案の文字列配列です。")
            require(bool(condition["missing"]) if condition["status"] == "判定保留" else bool(condition["refs"]), "保留にはmissing、充足・不充足にはrefsが必要です。")
        require(set(ids) == {"C1", "C2", "C3", "C4"}, "C1〜C4を重複なく一度ずつ指定してください。")


def validate_visible_refs(state, refs):
    for ref in refs:
        document = ref["document"]
        require(document in state["observed"], "根拠資料をまだ観測していません: " + document)
        available = complete_lines(document_text(state, document), state["observed"][document]["spans"])
        absent = [line for line in ref["lines"] if line not in available]
        require(not absent, "未観測または一部分しか観測していない根拠行です: {} {}".format(document, absent))


def execute_action(state, action):
    kind = action["type"]
    if kind == "search":
        collection = action["collection"]
        if not state["collections"][collection]:
            return {"status": "ok", "type": kind, "retrieved_count": 0, "results": [], "note": "指定collectionで該当資料を取得できませんでした。資料そのものの不存在を意味しません。"}
        try:
            index = load_index(learner_path(state["collection_paths"][collection]), 400, 0)
            selected, report = retrieve(index, action["query"], action["top_k"])
        except RetrievalError as exc:
            raise ActionError(str(exc)) from None
        results = []
        for rank, (score, chunk) in enumerate(selected, 1):
            document = Path(chunk["source"]).name
            raw = document_text(state, document)
            results.append({"rank": rank, "score": score, **observe(state, document, collection + "/" + chunk["source"], raw, chunk["start_offset"], chunk["end_offset"])})
        return {"status": "ok", "type": kind, "retrieved_count": len(results), "results": results,
                "note": "検索スコアは字面の類似度であり正しさではありません。固定L番号に触れていても完全行とは限りません。0件は資料の不存在を意味しません。"}
    if kind == "read":
        document, collection = action["document"], action["collection"]
        require(document in state["collections"][collection], "指定collection内では対象資料を取得できません: " + document)
        raw = document_text(state, document)
        if "sections" in action:
            require(document in state["structure"], "この資料の条構造が定義されていません。全文取得を検討してください。")
            keys = ["article" + str(n) for n in action["sections"]]
            require(all(key in state["structure"][document] for key in keys), "指定した条が構造マップにありません。")
            ranges = sorted((state["structure"][document][key]["start_offset"], state["structure"][document][key]["end_offset"], key) for key in keys)
        else:
            ranges = [(0, len(raw), "full_document")]
        return {"status": "ok", "type": kind, "results": [{"section": section, **observe(state, document, collection + "/" + document, raw, start, end)} for start, end, section in ranges]}
    if kind == "compare":
        validate_visible_refs(state, action["refs"])
        values = []
        for value in (action["left"], action["right"]):
            if action["value_type"] == "number":
                require(type(value) in (int, float, str), "numberの値は数値または数値文字列で指定してください。")
                try:
                    require(bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(value))), "numberに通貨記号や桁区切りは使えません。")
                    parsed = Decimal(str(value))
                    require(parsed.is_finite(), "numberは有限数にしてください。")
                except InvalidOperation:
                    raise ActionError("numberの数値表現が不正です。") from None
            elif action["value_type"] == "datetime":
                require(isinstance(value, str), "datetimeはタイムゾーン付きISO8601文字列です。")
                try:
                    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
                except ValueError:
                    raise ActionError("datetimeの形式が不正です。") from None
                require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "datetimeにはタイムゾーンが必要です。")
            else:
                require(isinstance(value, str), "textは文字列を指定してください。")
                parsed = value
            values.append(parsed)
        left, right = values
        result = left < right if action["op"] == "lt" else left <= right if action["op"] == "le" else left == right
        return {"status": "ok", "type": kind, "result": result, "op": action["op"], "value_type": action["value_type"],
                "left": action["left"], "right": action["right"], "refs": action["refs"],
                "note": "渡された値を決定的に比較しました。根拠行の実在・完全行の観測を確認しましたが、値の抽出の正しさや規程・条件への適用は保証しません。"}
    for condition in action["conditions"]:
        validate_visible_refs(state, condition["refs"])
    state["final_result"] = {"conditions": action["conditions"], "summary": action["summary"]}
    state["status"] = "finished"
    return {"status": "ok", "type": "finish", "accepted": True,
            "note": "4条件の提出を受け付けました。根拠行の観測は検査しましたが、判定内容や規程適用の正誤は評価していません。人がレビューしてください。"}


def step(folder, action_path):
    state = load_session(folder)
    if state["status"] != "active":
        raise LabError("終了済みのsessionは追加stepを受け付けません: " + state["status"])
    path = action_path.resolve()
    if not inside(path, WORKSPACE) or path.suffix.lower() != ".json" or any(part.lower() in {"teacher", "instructor"} for part in path.parts):
        raise LabError("actionはworkspace内の講師用資料以外のJSONファイルにしてください。")
    action = load_json(path)
    try:
        validate_action(action)
        fingerprint = digest(encoded({key: value for key, value in action.items() if key != "purpose"}))
        if fingerprint in state["fingerprints"]:
            observation = {"status": "duplicate", "note": "同じ引数の行動は実行しませんでした。取得済み情報を確認し、別の引数・行動またはfinishを選んでください。"}
        else:
            state["fingerprints"].append(fingerprint)
            observation = execute_action(state, action)
    except ActionError as exc:
        observation = {"status": "error", "message": str(exc), "note": "この試行は手数に数えます。引数を修正するか別の行動を選んでください。"}
    state["step_count"] += 1
    if state["status"] != "finished" and state["step_count"] >= state["max_steps"]:
        state["status"] = "halted_budget"
    state["history"].append({"step": state["step_count"], "action": action, "observation": observation, "status_after": state["status"]})
    save_session(folder, state)
    print("step {}: {} / session {}".format(state["step_count"], observation["status"], state["status"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="新しい手動中継sessionを作る")
    init.add_argument("--scenario", required=True)
    init.add_argument("--out", required=True)
    submit = commands.add_parser("step", help="人が保存した行動JSONを1回処理する")
    submit.add_argument("--session", required=True)
    submit.add_argument("--action", required=True, type=Path)
    show = commands.add_parser("show", help="整合性を確認し、中継用ファイルのパスを表示する")
    show.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            folder = session_path(args.out)
            if folder.exists():
                raise LabError("既存のsessionは初期化できません。新しい名前を指定してください。")
            state = initial_state(learner_path(args.scenario))
            save_session(folder, state, create=True)
            print("初期化しました: " + relative(folder) + " / モデル呼び出しなし")
        elif args.command == "step":
            step(session_path(args.session), args.action)
        else:
            folder = session_path(args.session)
            load_session(folder)
            for name in OUTPUTS:
                print(relative(folder / name))
        return 0
    except (LabError, OSError) as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
