"""教材の検索条件を実行する。LLMへの接続や課金は行わない。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='実行する条件のID。省略時は全条件')
    parser.add_argument('--overwrite', action='store_true', help='既存の検索結果を意図して更新する')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    experiments = json.loads((root / 'learner/experiments.json').read_text(encoding='utf-8'))
    chosen = [x for x in experiments if not args.only or x['id'] == args.only]
    if not chosen:
        parser.error('指定された条件IDがありません。learner/experiments.jsonを確認してください。')
    for ex in chosen:
        identifier = ex['id']
        if not identifier or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identifier):
            parser.error('条件IDには英数字、ハイフン、アンダースコアを使ってください。')
        command = [sys.executable, str(root/'retrieve.py'), '--corpus', ex['corpus'], '--base', ex['base'],
                   '--query', ex['query'], '--chunk-size', str(ex['chunk_size']), '--top-k', str(ex['top_k']),
                   '--overlap', str(ex['overlap']), '--out', 'runs/' + identifier]
        if args.overwrite:
            command.append('--overwrite')
        print(identifier, flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            return result.returncode
    print('検索結果とLLM入力文をrunsに保存しました。LLMは呼び出していません。')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
