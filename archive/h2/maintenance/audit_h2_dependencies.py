"""Read-only H2 dependency inventory. Run at repository root; no fits or moves."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import re
from pathlib import Path

ENTRY = ('src/features/h2.py', 'src/h2/models/specialist.py',
         'src/h2/models/regime.py', 'src/h2/models/blend.py',
         'src/h2/evaluation/reproduce_specialist.py', 'src/h2/pipeline.py')
EVIDENCE = {'audit_h2_leakage', 'audit_h2_source_pit', 'evaluate_h2_pit_robustness',
            'evaluate_h2_overfitting', 'evaluate_h2_monetary_regime_robustness',
            'evaluate_h2_rate_hike_robustness'}
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.worktrees', '.pytest_cache'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    a = ap.parse_args(); root = a.root.resolve()
    if not all((root / s).is_file() for s in ENTRY):
        ap.error('Missing entrypoint(s); run from current finance_analytics root')
    out = root / 'outputs/h2_code_cleanup/dependency_review.csv'
    src = sorted(p for p in (root/'src').rglob('*.py') if not set(p.parts)&SKIP)
    tests = sorted(root.glob('test*.py')) + sorted((root/'tests').rglob('*.py'))
    files = sorted(set(src+tests))
    exp = sorted((root/'src/h2/experiments').glob('*.py'))
    targets = {p.stem:p.relative_to(root).as_posix() for p in exp if p.stem!='__init__'}
    if not targets: ap.error('No experiments found')
    paths = {p.relative_to(root).as_posix(): p for p in files}
    names = {}
    for rel in paths:
        names.setdefault(Path(rel).stem, []).append(rel)
    refs = {t:set() for t in targets.values()}; graph = {p:set() for p in paths}
    errors=[]
    for rel,p in paths.items():
        try:
            source = p.read_text(encoding='utf-8-sig'); tree=ast.parse(source,filename=rel)
        except (OSError,UnicodeError,SyntaxError) as exc:
            errors.append(f'{rel}: {exc}'); continue
        # Conservative stem matching inside import statements, including local imports.
        # May over-report when stems are shared across packages.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                items=[x.name for x in node.names]
                if isinstance(node, ast.ImportFrom) and node.module: items.append(node.module)
                for item in items:
                    for part in item.split('.'):
                        for dest in names.get(part,[]):
                            if dest!=rel: graph[rel].add(dest)
        # Also catch non-import textual references (dynamic imports, paths, etc.).
        for stem,dest in targets.items():
            if rel!=dest and re.search(r'(?<!\w)'+re.escape(stem)+r'(?!\w)',source):
                refs[dest].add(rel)
    reachable=set(ENTRY); todo=list(ENTRY)
    while todo:
        for dest in graph.get(todo.pop(),()):
            if dest not in reachable: reachable.add(dest); todo.append(dest)
    audit_refs={t:set() for t in targets.values()}
    for folder in ('outputs','results','data/processed'):
        for p in (root/folder).rglob('*.json'):
            if not any(w in p.name.lower() for w in ('audit','manifest','metadata','config')): continue
            try: source=p.read_text(encoding='utf-8-sig')
            except (OSError,UnicodeError) as exc:
                errors.append(f'{p.relative_to(root)}: {exc}'); continue
            for stem,dest in targets.items():
                if re.search(r'(?<!\w)'+re.escape(stem)+r'(?!\w)',source):
                    audit_refs[dest].add(p.relative_to(root).as_posix())
    rows=[]
    for stem,rel in sorted(targets.items()):
        imported=sorted(p for p,edges in graph.items() if rel in edges)
        if rel in reachable: status='KEEP_ENTRY_DEPENDENCY'
        elif stem in EVIDENCE: status='KEEP_RESEARCH_EVIDENCE'
        elif imported or refs[rel] or audit_refs[rel]: status='KEEP_REFERENCED'
        else: status='REVIEW_ONLY_NO_MOVE'
        rows.append(dict(path=rel,status=status,sha256=digest(root/rel),
                         import_references='; '.join(imported),source_references='; '.join(sorted(refs[rel])),
                         audit_references='; '.join(sorted(audit_refs[rel]))))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r['status'],r['path'])
    print('Scanned Python files:',len(files),'| experiments:',len(rows),'| scan errors:',len(errors))
    for err in errors: print('SCAN ERROR:',err)
    print('Saved:',out)
    print('PREVIEW ONLY: no code/data moved; static scan is not proof of runtime independence.')
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
