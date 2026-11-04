#!/usr/bin/env python3
"""文字bigramのTF-IDFで資料を検索し、LLMへ渡す入力文を準備する。

Python 3の標準ライブラリだけで動作する。モデル、API、外部検索は使用しない。
ベクトルは字面の出現頻度から計算する疎ベクトルであり、学習済みモデルによる
パラメトリックな埋め込みではない。検索結果をbase本文へ付加するまでを行い、
回答や監査上の評価は生成しない。

パスは現在の作業フォルダーを基準に指定する。文字位置は、UTF-8を復号した
原文のUnicodeコードポイント数で数え、開始を含み、終了を含まない。
改行や正規化前の文字も位置に含む。検索用の正規化は原文自体を変更しない。
"""

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


OUTPUT_NAMES = ("index.json", "retrieval.json", "retrieved.txt", "prompt.txt")
LINE_LABEL = re.compile(r"^\s*(?:\[(L\d+)\]|(L\d+))(?=\s|[：:|｜.]|$)")
METHOD = {
    "name": "character_bigram_tfidf",
    "description": "字面由来の疎ベクトル。学習済みモデルによる埋め込みではない。",
    "normalization": "NFKC、casefold、Unicode空白の除去。句読点は保持。",
    "term": "正規化後の隣接する2文字。1文字だけではtermを作らない。",
    "tf": "termの出現回数",
    "idf": "ln((1 + N) / (1 + df)) + 1。Nはchunk数、dfはtermを含むchunk数。",
    "normalization_of_vector": "TF-IDF値をL2正規化",
    "similarity": "cosine。語彙外のquery termは無視する。",
    "tie_break": "相対資料パス、原文の開始文字位置の順",
}


class RetrievalError(Exception):
    """検索入力または保存先の問題により処理を続けられない。"""


def normalize(text):
    """検索専用の文字列を作る。原文の位置情報には使用しない。"""
    return "".join(
        char for char in unicodedata.normalize("NFKC", text).casefold()
        if not char.isspace()
    )


def bigrams(text):
    normalized = normalize(text)
    return Counter(normalized[i:i + 2] for i in range(len(normalized) - 1))


def read_utf8(path, label):
    """改行コードを置換せずに読み、原文の文字位置を維持する。"""
    try:
        if not path.is_file():
            raise RetrievalError("{}: 資料ファイルがありません。".format(label))
        raw = path.read_bytes()
        return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    except UnicodeError:
        raise RetrievalError("{}: UTF-8で読み取れません。".format(label)) from None
    except OSError:
        raise RetrievalError("{}: 資料を読み取れません。".format(label)) from None


def within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def original_lines(text):
    """原文の各行に、物理行番号・任意の固定L番号・文字位置を付ける。"""
    lines = []
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        match = LINE_LABEL.match(line)
        label_group = (1 if match.group(1) else 2) if match else None
        lines.append({
            "source_line": number,
            "label": (match.group(1) or match.group(2)) if match else None,
            "label_start_offset": offset + match.start(label_group) if match else None,
            "label_end_offset": offset + match.end(label_group) if match else None,
            "start_offset": offset,
            "end_offset": offset + len(line),
        })
        offset += len(line)
    return lines


def split_document(source, text, chunk_size, overlap):
    """文書を跨がず固定文字数で分割し、断片が実際に覆う行を記録する。"""
    lines = original_lines(text)
    starts = [line["start_offset"] for line in lines]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        covered = []
        line_index = max(0, bisect_right(starts, start) - 1)
        for line in lines[line_index:]:
            if line["start_offset"] >= end:
                break
            if line["end_offset"] > start:
                covered.append({
                    "source_line": line["source_line"],
                    "label": line["label"],
                    "label_start_offset": line["label_start_offset"],
                    "label_end_offset": line["label_end_offset"],
                    "start_offset": max(start, line["start_offset"]),
                    "end_offset": min(end, line["end_offset"]),
                })
        excerpt = text[start:end]
        chunks.append({
            "source": source,
            "start_offset": start,
            "end_offset": end,
            "source_start_line": covered[0]["source_line"],
            "source_end_line": covered[-1]["source_line"],
            "starts_mid_line": start > lines[line_index]["start_offset"],
            "ends_mid_line": end < lines[covered[-1]["source_line"] - 1]["end_offset"],
            "lines": covered,
            "text": excerpt,
            "terms": dict(sorted(bigrams(excerpt).items())),
        })
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def normalized_vector(terms, idf):
    weighted = {term: count * idf[term] for term, count in sorted(terms.items()) if term in idf}
    norm = math.sqrt(math.fsum(value * value for value in weighted.values()))
    if not norm:
        return {}
    return {term: value / norm for term, value in weighted.items()}


def load_index(corpus, chunk_size, overlap):
    try:
        if not corpus.is_dir():
            raise RetrievalError("corpus: 資料フォルダーがありません。")
        paths = sorted(
            (p for p in corpus.rglob("*.txt") if p.is_file()),
            key=lambda p: p.relative_to(corpus).as_posix(),
        )
    except OSError:
        raise RetrievalError("corpus: 資料一覧を取得できません。") from None
    if not paths:
        raise RetrievalError("corpus: UTF-8の.txt資料がありません。")

    documents = []
    chunks = []
    has_content = False
    for path in paths:
        source = path.relative_to(corpus).as_posix()
        text, digest = read_utf8(path, source)
        has_content = has_content or bool(text.strip())
        documents.append({"source": source, "characters": len(text), "sha256": digest})
        chunks.extend(split_document(source, text, chunk_size, overlap))
    if not has_content:
        raise RetrievalError("corpus: 資料の本文がすべて空です。")

    df = Counter()
    for chunk in chunks:
        df.update(chunk["terms"].keys())
    count = len(chunks)
    idf = {term: math.log((1 + count) / (1 + frequency)) + 1 for term, frequency in sorted(df.items())}
    for number, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = "chunk-{:06d}".format(number)
        chunk["vector"] = normalized_vector(chunk["terms"], idf)
    return {
        "schema_version": 1,
        "method": METHOD,
        "chunking": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "offset_unit": "原文のUnicodeコードポイント。開始を含み、終了を含まない。",
            "line_reference": "原文の物理行番号と、存在する場合は固定L番号。行の一部分だけを含むこともある。",
        },
        "documents": documents,
        "chunk_count": count,
        "idf": idf,
        "chunks": chunks,
    }


def retrieve(index, query, top_k):
    query_terms = bigrams(query)
    vector = normalized_vector(query_terms, index["idf"])
    ranked = []
    for chunk in index["chunks"]:
        score = math.fsum(value * chunk["vector"].get(term, 0.0) for term, value in vector.items())
        # 浮動小数点の丸めだけで1を僅かに超えないようにする。
        score = max(0.0, min(1.0, score))
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["start_offset"]))
    selected = [(score, chunk) for score, chunk in ranked if score > 0][:top_k]
    selected_ids = {chunk["chunk_id"] for _, chunk in selected}
    ranking = []
    for rank, (score, chunk) in enumerate(ranked, start=1):
        ranking.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "start_offset": chunk["start_offset"],
            "end_offset": chunk["end_offset"],
            "source_start_line": chunk["source_start_line"],
            "source_end_line": chunk["source_end_line"],
            "starts_mid_line": chunk["starts_mid_line"],
            "ends_mid_line": chunk["ends_mid_line"],
            "lines": chunk["lines"],
            "score": score,
            "selected": chunk["chunk_id"] in selected_ids,
        })
    report = {
        "schema_version": 1,
        "query": query,
        "normalized_query": normalize(query),
        "parameters": {
            "chunk_size": index["chunking"]["chunk_size"],
            "overlap": index["chunking"]["overlap"],
            "top_k": top_k,
        },
        "method": METHOD["name"],
        "query_terms": dict(sorted(query_terms.items())),
        "query_vector": vector,
        "candidate_count": len(ranked),
        "positive_score_count": sum(score > 0 for score, _ in ranked),
        "retrieved_count": len(selected),
        "selected_chunk_ids": [chunk["chunk_id"] for _, chunk in selected],
        "ranking": ranking,
    }
    return selected, report


def retrieved_text(selected):
    parts = [
        "検索方式: 文字bigramのTF-IDFとcosine（字面由来の疎ベクトル）\n",
        "取得件数: {}\n".format(len(selected)),
        "文字位置は開始を含み終了を含まない。行の一部分だけを含む場合がある。\n",
    ]
    if not selected:
        parts.append("\n一致するbigramを持つ検索結果はありません。資料の不存在を示すものではありません。\n")
    for rank, (score, chunk) in enumerate(selected, start=1):
        labels = list(dict.fromkeys(line["label"] for line in chunk["lines"] if line["label"]))
        parts.extend([
            "\n[取得 {} / {}]\n".format(rank, chunk["chunk_id"]),
            "資料: {}\n".format(chunk["source"]),
            "原文文字位置: [{}, {})\n".format(chunk["start_offset"], chunk["end_offset"]),
            "原文行: {}-{}\n".format(chunk["source_start_line"], chunk["source_end_line"]),
            "行途中の開始 / 終了: {} / {}\n".format(
                "あり" if chunk["starts_mid_line"] else "なし",
                "あり" if chunk["ends_mid_line"] else "なし",
            ),
            "固定行番号: {}\n".format(", ".join(labels) if labels else "なし（原文行番号を参照）"),
            "検索スコア: {:.12g}\n".format(score),
            "--- 原文断片 ---\n",
            chunk["text"],
            "\n--- 原文断片終わり ---\n",
        ])
    return "".join(parts)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_outputs(out, corpus, base, outputs, overwrite):
    """全既存出力を先に比較し、不一致時は何も書き込まず停止する。"""
    if out.exists() and not out.is_dir():
        raise RetrievalError("out: 保存先がフォルダーではありません。")
    states = {}
    differences = []
    for name in OUTPUT_NAMES:
        path = out / name
        resolved = path.resolve()
        if within(resolved, corpus) or resolved == base or path.is_symlink():
            raise RetrievalError("{}: 入力資料またはリンク先への書き込みはできません。".format(name))
        if not path.exists():
            states[name] = "作成"
        elif not path.is_file():
            raise RetrievalError("{}: 同名のファイル以外の項目があります。".format(name))
        elif path.read_bytes() == outputs[name]:
            states[name] = "一致"
        else:
            states[name] = "更新"
            differences.append(name)
    if differences and not overwrite:
        raise RetrievalError(
            "既存内容と異なります: {}。書き込みは行っていません。\n"
            "別の --out を使うか、意図して再構成する場合は --overwrite を指定してください。"
            .format(", ".join(differences))
        )

    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if states[name] != "一致":
            try:
                (out / name).write_bytes(outputs[name])
            except OSError:
                raise RetrievalError(
                    "{}: 保存できません。保存済みファイルがある場合は、原因を解消して再実行してください。"
                    .format(name)
                ) from None
        print("{}: {}".format(name, states[name]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path, help="UTF-8の.txt資料を再帰検索するフォルダー")
    parser.add_argument("--base", required=True, type=Path, help="検索結果を付加するUTF-8入力本文")
    parser.add_argument("--query", required=True, help="検索文。正規化後に空の指定は不可")
    parser.add_argument("--chunk-size", type=int, default=600, help="原文のUnicode文字数（既定: 600）")
    parser.add_argument("--top-k", type=int, default=1, help="正のスコアから取得する最大件数（既定: 1）")
    parser.add_argument("--overlap", type=int, default=0, help="隣接chunkの重なり文字数（既定: 0）")
    parser.add_argument("--out", required=True, type=Path, help="4つの出力を保存するフォルダー。corpus配下は不可")
    parser.add_argument("--overwrite", action="store_true", help="既存結果と異なる場合も、意図して再構成する")
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size は1以上で指定してください。")
    if args.top_k < 1:
        parser.error("--top-k は1以上で指定してください。")
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap は0以上、--chunk-size 未満で指定してください。")
    if not normalize(args.query):
        parser.error("--query は空白以外の文字を含めてください。")

    try:
        corpus = args.corpus.resolve()
        base = args.base.resolve()
        out = args.out.resolve()
        if within(out, corpus):
            raise RetrievalError("out: corpusと同じ場所や、その配下には保存できません。")
        base_text, base_hash = read_utf8(base, "base")
        if not base_text.strip():
            raise RetrievalError("base: 入力本文が空です。")
        index = load_index(corpus, args.chunk_size, args.overlap)
        selected, report = retrieve(index, args.query, args.top_k)
        report["base_sha256"] = base_hash
        text = retrieved_text(selected)
        # baseを変更せず、その後ろに検索結果だけを追加する。
        prompt = base_text + "\n\n【検索結果】\n" + text
        outputs = {
            "index.json": json_bytes(index),
            "retrieval.json": json_bytes(report),
            "retrieved.txt": text.encode("utf-8"),
            "prompt.txt": prompt.encode("utf-8"),
        }
        save_outputs(out, corpus, base, outputs, args.overwrite)
        print("取得 {} 件 / 全 {} chunk（LLMの呼び出しなし）".format(len(selected), index["chunk_count"]))
        return 0
    except RetrievalError as exc:
        print("エラー: {}".format(exc), file=sys.stderr)
        return 1
    except OSError:
        print("エラー: 資料または保存先にアクセスできません。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
