"""Conservative H2 source inventory and explicitly selected archival (no model/data access).
Run at repository root: python organize_h2_code.py
Apply only reviewed names: python organize_h2_code.py --apply --select FILE.py
Archived scripts are historical source snapshots; restore original paths to rerun them.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

# Exact names only; unknown/new files are deliberately not candidates.
CANDIDATES = {
    'run_h2_debt_quartile.py', 'run_h2_dual_results.py',
    'run_h2_factorial_models.py', 'run_h2_feature_upgrade.py',
    'run_h2_full8_integrated.py', 'run_h2_full8_only.py',
    'run_h2_integrated_ablation.py', 'run_h2_only_ablation.py',
    'run_h2_paper_models.py', 'run_h2_ratechange_ablation.py',
    'evaluate_h2_integrated_ablation.py', 'evaluate_h2_rate_hike_robustness.py',
    'collect_h2_pit_evidence.py',
}
PROTECTED = {
    '_h2_full8_shared.py', 'build_h2_debt_change.py', 'recover_h2_missing_features.py',
    'evaluate_h2_training_period_and_features.py', 'evaluate_h2_conditional_full_model.py',
    'evaluate_h2_regime_specialized_model.py', 'evaluate_h2_independent_regime_model.py',
    'evaluate_h2_independent_robustness.py', 'evaluate_h2_exante_universe.py',
    'evaluate_h2_pit_robustness.py', 'data_io.py', 'pipeline.py', '__init__.py',
}
TEXT = {'.py','.md','.rst','.txt','.json','.yaml','.yml','.toml','.ini','.cfg','.ps1','.sh','.bat','.cmd','.ipynb'}
SKIP = {'.git','.venv','venv','__pycache__','.pytest_cache','node_modules','.mypy_cache','.ruff_cache'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan_paths(root, out, script):
    """Git-visible code/docs/config + local source/config + historical audit JSON."""
    paths = set()
    git = subprocess.run(['git','-C',str(root),'ls-files','-z','--cached','--others','--exclude-standard'],
                         capture_output=True, check=True)
    for raw in git.stdout.decode('utf-8',errors='strict').split('\0'):
        if not raw: continue
        p=root/raw
        if p.suffix.lower() in TEXT and not set(Path(raw).parts)&SKIP and Path(raw).parts[0] not in {'data','outputs','results'}:
            paths.add(p)
    for folder in ['src','config','scripts','tests','docs','archive']:
        for p in (root/folder).rglob('*'):
            if p.suffix.lower() in TEXT and not set(p.relative_to(root).parts)&SKIP:
                paths.add(p)
    for folder in ['outputs','results','data/processed']:
        for p in (root/folder).rglob('*.json'):
            if any(word in p.name.lower() for word in ('audit','manifest','metadata','config')):
                paths.add(p)
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in TEXT: paths.add(p)
    return sorted(p for p in paths if p.is_file() and p.resolve()!=script and out not in p.resolve().parents)


def inspect(root,out):
    h2=root/'src/h2'; exp=h2/'experiments'; script=Path(__file__).resolve()
    files=sorted(p for p in h2.rglob('*.py') if '__pycache__' not in p.parts)
    texts={}; errors=[]
    for p in scan_paths(root,out,script):
        try:
            if not p.resolve().is_relative_to(root) or p.is_symlink():
                raise ValueError('symlink or outside repository')
            text=p.read_text(encoding='utf-8-sig')
            # Syntax issues in active Python files are blockers to automatic archival.
            if p.suffix=='.py' and p.is_relative_to(root/'src'): ast.parse(text,filename=str(p))
            texts[p]=text
        except (OSError, UnicodeError, SyntaxError, ValueError) as e:
            errors.append({'path':p.relative_to(root).as_posix(),'error':str(e)})
    rows=[]
    for p in files:
        rel=p.relative_to(root).as_posix(); name=p.name
        # Token matching deliberately catches imports, module commands, docs and path/hash references.
        pattern=re.compile(r'(?<![\w])'+re.escape(p.stem)+r'(?![\w])')
        refs=[]
        if p.parent==exp and name in CANDIDATES:
            for other,text in texts.items():
                if other==p: continue
                if pattern.search(text): refs.append(other.relative_to(root).as_posix())
        dest=root/'archive/h2/experiments'/name
        if p.parent!=exp or name in PROTECTED:
            status='KEEP_CORE_OR_DEPENDENCY'
        elif name not in CANDIDATES:
            status='KEEP_UNREVIEWED'
        elif refs:
            status='KEEP_REFERENCED'
        elif dest.exists() or dest.is_symlink():
            status='KEEP_DESTINATION_EXISTS'
        elif errors or p.is_symlink() or not p.resolve().is_relative_to(root):
            status='KEEP_SCAN_BLOCKED'
        else:
            status='ARCHIVE_CANDIDATE'
        rows.append({'path':rel,'status':status,'sha256':sha(p) if p.is_file() and not p.is_symlink() else '',
                     'references':refs,'destination':dest.relative_to(root).as_posix() if status=='ARCHIVE_CANDIDATE' else ''})
    return rows,errors,len(texts)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('outputs/h2_code_cleanup'))
    ap.add_argument('--apply',action='store_true',help='Move explicitly selected, currently unreferenced candidates only')
    ap.add_argument('--select',nargs='+',default=[],metavar='FILE.py')
    a=ap.parse_args();root=a.root.resolve();out=(root/a.out).resolve()
    if not (root/'src/h2/experiments').is_dir() or not (root/'config/experiment.yaml').is_file():
        ap.error('Run at the finance_analytics repository root')
    if not out.is_relative_to(root/'outputs') or out==root/'outputs': ap.error('--out must be a subfolder of outputs')
    if a.apply != bool(a.select): ap.error('--apply and --select must be supplied together')
    if len(a.select)!=len(set(a.select)) or any(Path(s).name!=s for s in a.select): ap.error('Select distinct file basenames only')
    rows,errors,n=inspect(root,out)
    eligible={Path(r['path']).name:r for r in rows if r['status']=='ARCHIVE_CANDIDATE'}
    plan={'mode':'apply' if a.apply else 'preview','scanned_text_files':n,'scan_errors':errors,'files':rows,'moved':[],
          'limitations':'Conservative static token scan, not proof of runtime dependency completeness; no model/data evaluation. Restore archived scripts to original paths for historical reproduction.'}
    out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'h2_code_inventory.csv'; json_path=out/'h2_archive_plan.json'
    # Preserve previous apply audit rather than overwrite it.
    if a.apply and (out/'h2_archive_applied.json').exists(): ap.error('Apply audit already exists; use a new --out folder')
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','status','sha256','references','destination']);w.writeheader()
        for r in rows: w.writerow({**r,'references':'; '.join(r['references'])})
    json_path.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
    for s in sorted(set(r['status'] for r in rows)):
        print(f'{s}: {sum(r["status"]==s for r in rows)}')
    for name in sorted(eligible): print('  CANDIDATE:',name)
    if errors: print('SCAN BLOCKED:',len(errors),'file(s); no archival allowed')
    if a.apply:
        if errors or not set(a.select)<=set(eligible): ap.error('Selection contains a blocked/unknown file; inspect inventory')
        # No overwrite, no modification of existing source contents, no git staging.
        destdir=root/'archive/h2/experiments'
        if destdir.resolve()!=destdir: ap.error('Archive directory must not be a symlink')
        destdir.mkdir(parents=True,exist_ok=True)
        for name in a.select:
            r=eligible[name];p=root/r['path'];dest=root/r['destination']
            if dest.exists() or dest.is_symlink() or sha(p)!=r['sha256']: ap.error(f'File changed since scan: {name}')
        applied=out/'h2_archive_applied.json'
        applied.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
        for name in a.select:
            r=eligible[name];p=root/r['path'];dest=root/r['destination']
            # Exclusive target creation prevents a race from overwriting a file.
            with dest.open('xb') as f: f.write(p.read_bytes())
            if sha(dest)!=r['sha256'] or sha(p)!=r['sha256']:
                raise RuntimeError(f'Copy verification failed; original retained: {name}')
            p.unlink()
            plan['moved'].append({'from':r['path'],'to':r['destination'],'sha256':r['sha256']})
            applied.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
            print('MOVED:',name)
    print('Saved:',csv_path,json_path)
    if not a.apply: print('PREVIEW ONLY: no source/data files moved or edited')

if __name__=='__main__': main()
