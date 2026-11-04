"""実習編3の検索と文脈取得を実行する。LLMには接続しない。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='実行する条件ID。省略時は全7条件')
    parser.add_argument('--overwrite', action='store_true', help='既存結果を意図して更新する')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    experiments = json.loads((root/'learner/experiments.json').read_text(encoding='utf-8'))
    chosen = [e for e in experiments if not args.only or e['id'] == args.only]
    if not chosen:
        parser.error('指定された条件IDがありません。learner/experiments.jsonを確認してください。')
    for ex in chosen:
        identifier = ex['id']
        if not identifier or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identifier):
            parser.error('条件IDには英数字、ハイフン、アンダースコアを使ってください。')
        script = 'basic_retrieve.py' if ex['mode'] == 'basic' else 'advanced_retrieve.py'
        command = [sys.executable, str(root/script), '--corpus', ex['corpus'], '--base', ex['base'],
            '--query', ex['query'], '--chunk-size', str(ex['chunk_size']), '--overlap', str(ex['overlap']),
            '--top-k', str(ex['top_k']), '--out', 'runs/'+identifier]
        if ex['mode'] != 'basic':
            command += ['--mode', ex['mode'], '--structure', ex['structure'], '--max-sections', str(ex['max_sections'])]
        if args.overwrite:
            command.append('--overwrite')
        print(identifier, flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            return result.returncode
    print('検索と文脈取得の結果をrunsに保存しました。LLMは呼び出していません。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
