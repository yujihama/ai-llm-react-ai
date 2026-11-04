"""講師用の事前記述した行動例を再生する。LLMの行動選択は行わない。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['C_same','C_deferred','C_unavailable'])
    parser.add_argument('--prefix',default='scripted',help='runs内の新規フォルダー名の接頭辞')
    args=parser.parse_args()
    if not args.prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.prefix):
        parser.error('prefixは英数字、ハイフン、アンダースコアだけにしてください。')
    root=Path(__file__).resolve().parent
    names=[args.only] if args.only else ['C_same','C_deferred','C_unavailable']
    for name in names:
        scenario='learner/scenarios/'+name+'.json'
        session='runs/'+args.prefix+'_'+name
        run=root/session
        if run.exists():
            parser.error('既存の実行は上書きしません。別のprefixを使ってください。')
        subprocess.run([sys.executable,str(root/'react_lab.py'),'init','--scenario',scenario,'--out',session],cwd=root,check=True)
        data=json.loads((root/'instructor/example_actions'/f'{name}.json').read_text(encoding='utf-8'))
        (run/'example_provenance.txt').write_text(data['provenance']+'\n',encoding='utf-8')
        for i,action in enumerate(data['actions'],1):
            action_path=run/f'action-{i:02d}.json'
            action_path.write_text(json.dumps(action,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            subprocess.run([sys.executable,str(root/'react_lab.py'),'step','--session',session,'--action',str(action_path)],cwd=root,check=True)
    print('事前記述した行動例を再生しました。LLMの自律行動の実測ではありません。')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
