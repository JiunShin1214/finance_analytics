"""Archive only the seven reviewed H2 exploration scripts, as a group.
Run at repository root. Default is preview; --apply moves; --restore reverses.
Historical scripts may require restoration to their original import paths.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import tokenize
import ast
from pathlib import Path

NAMES = (
    '_h2_full8_shared.py', 'recover_h2_missing_features.py',
    'run_h2_factorial_models.py', 'run_h2_integrated_ablation.py',
    'run_h2_only_ablation.py', 'run_h2_paper_models.py',
    'run_h2_ratechange_ablation.py',
)
MAINTENANCE = {'organize_h2_code.py', 'audit_h2_dependencies.py',
               'extract_h2_training.py', 'upgrade_h2_to_8.py',
               'archive_h2_unused.py'}
SKIP = {'.git','.venv','venv','__pycache__','.worktrees','.pytest_cache'}
PATTERN = re.compile(r'(?<!\w)(' + '|'.join(re.escape(Path(n).stem) for n in NAMES) + r')(?!\w)')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan(root, selected):
    files = set(root.glob('*.py'))
    for folder in ('src','tests','scripts','config'):
        files.update(p for p in (root/folder).rglob('*') if p.suffix.lower() in {'.py','.yaml','.yml','.json','.ps1','.sh','.bat','.cmd','.ipynb'})
    for folder in ('outputs','results','data/processed'):
        files.update(p for p in (root/folder).rglob('*.json') if any(k in p.name.lower() for k in ('audit','manifest','metadata','config')) and not p.is_relative_to(root/'outputs/h2_code_cleanup'))
    blockers=[]; comments=[]; count=0
    for p in sorted(files):
        if p in selected or p.name in MAINTENANCE or set(p.relative_to(root).parts)&SKIP: continue
        rel=p.relative_to(root).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            blockers.append(f'{rel}: unsafe path'); continue
        try:
            s=p.read_text(encoding='utf-8-sig'); count+=1
            if p.suffix.lower()=='.py':
                ast.parse(s,filename=rel)
                toks=list(tokenize.generate_tokens(io.StringIO(s).readline))
                for t in toks:
                    for hit in PATTERN.findall(t.string):
                        msg=f'{rel}:{t.start[0]} -> {hit}'
                        (comments if t.type==tokenize.COMMENT else blockers).append(msg)
            else:
                for hit in sorted(set(PATTERN.findall(s))): blockers.append(f'{rel} -> {hit}')
        except (OSError,UnicodeError,SyntaxError,tokenize.TokenError) as e:
            blockers.append(f'{rel}: scan error: {e}')
    return count,sorted(set(blockers)),sorted(set(comments))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group(); modes.add_argument('--apply',action='store_true'); modes.add_argument('--restore',action='store_true')
    a=ap.parse_args(); root=Path.cwd().resolve()
    src=root/'src/h2/experiments'; dst=root/'archive/h2/experiments'
    audit=root/'outputs/h2_code_cleanup/archive_final.json'
    if not (root/'config/experiment.yaml').is_file() or not (root/'src/h2/evaluation/reproduce_specialist.py').is_file() or not src.is_dir(): ap.error('Run from current finance_analytics root')
    for p in (src,dst,audit.parent):
        if p.resolve()!=p: ap.error(f'Unsafe path: {p}')
    if a.restore:
        if not audit.is_file(): ap.error('Archive manifest missing')
        rows=json.loads(audit.read_text(encoding='utf-8'))['files']
        if {r['name'] for r in rows}!=set(NAMES) or len(rows)!=len(NAMES): ap.error('Unexpected manifest names')
        for r in rows:
            p=dst/r['name']; q=src/r['name']
            if not p.is_file() or p.is_symlink() or q.exists() or q.is_symlink() or sha(p)!=r['sha256']: ap.error(f'Restore blocked: {r["name"]}')
        for r in rows:
            (dst/r['name']).rename(src/r['name']); print('RESTORED:',r['name'])
        print('Restore complete. Manifest retained; do not reapply without reviewing it.')
        return
    selected={src/n for n in NAMES}
    for p in sorted(selected):
        q=dst/p.name
        if not p.is_file() or p.is_symlink() or q.exists() or q.is_symlink(): ap.error(f'Missing source or existing destination: {p.name}')
    count,blockers,comments=scan(root,selected)
    print('Scanned files:',count)
    for x in comments: print('COMMENT ONLY:',x)
    if blockers:
        for x in blockers: print('BLOCKED:',x)
        ap.error('External references or scan errors; no files moved.')
    rows=[{'name':n,'from':(src/n).relative_to(root).as_posix(),'to':(dst/n).relative_to(root).as_posix(),'sha256':sha(src/n)} for n in NAMES]
    for r in rows: print('CANDIDATE:',r['name'])
    if not a.apply:
        print('PREVIEW PASSED: 7 files; no changes. Run with --apply to move.')
        return
    if audit.exists(): ap.error('Existing archive_final.json; review previous operation first.')
    dst.mkdir(parents=True,exist_ok=True); audit.parent.mkdir(parents=True,exist_ok=True)
    for r in rows:
        if sha(src/r['name'])!=r['sha256'] or (dst/r['name']).exists(): ap.error('File changed during scan; no move.')
    with audit.open('x',encoding='utf-8') as f: json.dump({'status':'planned','files':rows,'scan_count':count,'comment_refs':comments},f,indent=2,ensure_ascii=False)
    moved=[]
    try:
        for r in rows:
            (src/r['name']).rename(dst/r['name']); moved.append(r)
            if sha(dst/r['name'])!=r['sha256']: raise RuntimeError('Hash mismatch')
            print('MOVED:',r['name'])
    except BaseException:
        for r in reversed(moved):
            if not (src/r['name']).exists(): (dst/r['name']).rename(src/r['name'])
        raise
    audit.write_text(json.dumps({'status':'completed','files':rows,'scan_count':count,'comment_refs':comments},indent=2,ensure_ascii=False),encoding='utf-8')
    print('ARCHIVE COMPLETE: 7 files; original hashes preserved.')
    print('Manifest:',audit)

if __name__=='__main__': main()
