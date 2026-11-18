"""Shared project paths and deterministic local I/O."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / 'data/processed/final'
VALIDATION = ROOT / 'results/validation'

def office_executable():
    """Resolve LibreOffice without a user-specific absolute path."""
    found=shutil.which('soffice.com') or shutil.which('soffice')
    if found:return Path(found)
    return Path(os.environ.get('ProgramFiles',''))/'LibreOffice/program/soffice.com'

def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def stock_ids(series):
    result = series.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)','',regex=True)
    numeric=result.str.fullmatch(r'\d{1,6}(?:\.0)?',na=False)
    result.loc[numeric]=result.loc[numeric].str.replace(r'\.0$','',regex=True).str.zfill(6)
    if result.isna().any() or not result.str.fullmatch(r'[0-9A-Z]{6}',na=False).all():
        raise ValueError('Invalid stock identifier')
    return result

def assert_unique(frame, columns):
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f'Missing or duplicate keys: {columns}')
