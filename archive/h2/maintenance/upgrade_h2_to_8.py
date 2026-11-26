"""Guarded local upgrade of src/features/h2.py. Run from finance_analytics root.
Default previews the source diff; --apply changes ONLY the Python source.
The upgraded builder still requires --execute --replace to replace h2.parquet.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import tempfile
from pathlib import Path

MARKER = '# H2_EIGHT_FEATURE_UPGRADE_V1'
ARCHIVE_SHA = '8a504b7d02f5e961fe07302cb33ae30db17e67fefd92273311cc08019a598df1'

CONSTANTS = '''# H2_EIGHT_FEATURE_UPGRADE_V1
FOUR_FEATURES = ['debt', 'policy', 'policy_x_debt', 'policy_change']
EXTRA_FEATURES = ['debt_change', 'rate_change', 'policy_change_x_debt', 'rate_change_x_debt']
MODEL_FEATURES = FOUR_FEATURES + EXTRA_FEATURES
FOUR_OUTPUT_COLUMNS = ['stock_id', 'base_month', 'eligible', 'debt_raw', *FOUR_FEATURES]
OUTPUT_COLUMNS = FOUR_OUTPUT_COLUMNS + EXTRA_FEATURES
REFERENCE_ARCHIVE = Path('data/processed/archive/h2_8features.parquet')
REFERENCE_SHA256 = '8a504b7d02f5e961fe07302cb33ae30db17e67fefd92273311cc08019a598df1'
'''
HELPERS = '''def extend_h2_eight(frame, debt_path):
    """Add four raw model features on the same base-month eligible universe.

    DebtChange is rebuilt from the exact Hybrid source used for debt. No targets,
    future months, archived feature values, ranking, or zero imputation are used.
    The first rate CSV month remains missing, as in the historical eight-feature run.
    """
    from src.h2.experiments.build_h2_debt_change import read_hybrid, compute_debt_change
    from src.h2.preprocessing.build_bok_rate_history import make

    source = read_hybrid(debt_path)
    # Never allow rows after the historical development input horizon to enter computation.
    source = source.loc[source.feature_month.le('2021-12')].copy()
    changes = compute_debt_change(source)
    changes = changes[['stock_id', 'feature_month', 'debt_final_raw', 'debt_change_raw']].rename(
        columns={'feature_month': 'base_month', 'debt_change_raw': 'debt_change'})
    out = frame.merge(changes, on=['stock_id', 'base_month'], how='left', validate='one_to_one')
    raw = out.proxy_masked.to_numpy(float)
    other = out.debt_final_raw.to_numpy(float)
    if not np.array_equal(np.isnan(raw), np.isnan(other)) or not np.allclose(raw, other, rtol=1e-12, atol=1e-10, equal_nan=True):
        raise ValueError('h2_debt_change_source_mismatch')
    out = out.drop(columns='debt_final_raw')
    _, rates = make()
    rates = rates[['feature_month', 'rate_change_pp']].sort_values('feature_month').copy()
    if rates.feature_month.duplicated().any() or rates.empty:
        raise ValueError('h2_invalid_rate_months')
    # Match run_h2_ratechange_ablation.add_rate_change: first file month forced NaN.
    rates.loc[rates.feature_month.eq(rates.feature_month.iloc[0]), 'rate_change_pp'] = np.nan
    rates = rates.rename(columns={'feature_month': 'base_month', 'rate_change_pp': 'rate_change'})
    if not out.base_month.isin(rates.base_month).all():
        raise ValueError('h2_rate_month_missing')
    out = out.merge(rates, on='base_month', how='left', validate='many_to_one')
    out['policy_change_x_debt'] = out.policy_change * out.debt
    out['rate_change_x_debt'] = out.rate_change * out.debt
    if np.isinf(out[MODEL_FEATURES].to_numpy(float)).any():
        raise ValueError('h2_infinite_feature')
    return out


def compare_feature_tables(candidate, reference, columns, *, atol=0.0, rtol=0.0):
    """Compare by keys, not storage row order. Report missingness and numeric drift."""
    key = ['stock_id', 'base_month']
    cols = key + [c for c in columns if c not in key]
    left, right = candidate[cols].copy(), reference[cols].copy()
    for table in (left, right):
        if table[key].isna().any().any():
            raise ValueError('h2_comparison_missing_keys')
        for c in key:
            table[c] = table[c].astype(str)
        if table.duplicated(key).any():
            raise ValueError('h2_comparison_duplicate_keys')
    left, right = (t.set_index(key).sort_index() for t in (left, right))
    same_keys = left.index.equals(right.index)
    result = {'keys_identical': bool(same_keys), 'candidate_rows': len(left),
              'reference_rows': len(right), 'candidate_only_keys': len(left.index.difference(right.index)),
              'reference_only_keys': len(right.index.difference(left.index)),
              'atol': atol, 'rtol': rtol, 'columns': {}}
    if same_keys:
        for c in cols[len(key):]:
            x, y = left[c], right[c]
            if c == 'eligible':
                bad = int((x.astype('string').fillna('<NA>') != y.astype('string').fillna('<NA>')).sum())
                result['columns'][c] = {'mismatches': bad}
                continue
            a, b = x.to_numpy(float), y.to_numpy(float)
            ma, mb = np.isnan(a), np.isnan(b)
            valid = ~ma & ~mb
            bad = int((~np.isclose(a[valid], b[valid], atol=atol, rtol=rtol)).sum())
            result['columns'][c] = {'missing_mismatches': int((ma ^ mb).sum()),
                                    'value_mismatches': bad,
                                    'max_abs_difference': float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None}
    result['match'] = bool(same_keys and all(
        not any(v.get(k, 0) for k in ('mismatches', 'missing_mismatches', 'value_mismatches'))
        for v in result['columns'].values()))
    return result


def check_eight_archive(root, out):
    """The frozen development archive is a reference only, never a build input."""
    path = root / REFERENCE_ARCHIVE
    if not path.is_file():
        raise FileNotFoundError(f'H2 historical reference missing: {path}')
    actual = sha256(path)
    if actual != REFERENCE_SHA256:
        raise ValueError(f'H2 historical reference SHA mismatch: {actual}')
    reference = pd.read_parquet(path)
    report = compare_feature_tables(out, reference, MODEL_FEATURES, atol=1e-10, rtol=1e-12)
    report['reference_sha256'] = actual
    report['reference_path'] = str(REFERENCE_ARCHIVE)
    return report


'''


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Expected exactly one source anchor; local version differs: ' + repr(old[:180]))
    return source.replace(old, new, 1)


def patch(source):
    if MARKER in source:
        raise ValueError('Upgrade marker already present; no changes made.')
    source = once(source, "MODEL_FEATURES = ['debt', 'policy', 'policy_x_debt', 'policy_change']\nOUTPUT_COLUMNS = ['stock_id', 'base_month', 'eligible', 'debt_raw', *MODEL_FEATURES]\nLEGACY_OUTPUT_COLUMNS = set(OUTPUT_COLUMNS) | {\n    'debt_change', 'rate_change', 'policy_change_x_debt', 'rate_change_x_debt',\n}\n", CONSTANTS)
    source = once(source, '2026-11-24 통합 실험에 따라 모델 입력 후보를 위 4개로 축소한다. 기존 8개 피처 버전은 별도 보존한다.',
        'H2 공식 특징표는 8개 피처를 생성한다. 기존 4개 열의 정의는 유지한다.\n- debt_change = 직전 다른 결산기 보고서 대비 부채비율 변화(원값).\n- rate_change = 기준월 말 정책금리 − 직전월 말 정책금리(%p), 첫 파일 월 결측.\n- policy_change_x_debt = policy_change × debt.\n- rate_change_x_debt = rate_change × debt.')
    source = once(source, '출력은 관리용 4개 열과 모델 피처 4개 열이다. 원천이 다르면 값이 달라질 수 있다.\n기존 8피처 파일에서 전환할 때 유지 열의 값과 결측 위치가 다르면 교체하지 않는다.',
        '출력은 관리용 4개 열과 모델 피처 8개 열이다. 원천이 다르면 교체를 중단한다.\n기존 4피처 파일 및 보존된 8피처 참조의 값·결측 위치를 검증한다.\nlag_6m 모드는 기존 4개 피처 점검만 허용하며 공식 출력은 Hybrid 8개 피처로 유지한다.')
    source = once(source, 'def build_h2(root: Path, config, *, debt_source:', HELPERS + 'def build_h2(root: Path, config, *, debt_source:')
    source = once(source, "    out = (f.rename(columns={'proxy_masked': 'debt_raw'})[OUTPUT_COLUMNS]", "    if debt_source == 'hybrid':\n        f = extend_h2_eight(f, debt_path)\n    output_columns = OUTPUT_COLUMNS if debt_source == 'hybrid' else FOUR_OUTPUT_COLUMNS\n    model_features = MODEL_FEATURES if debt_source == 'hybrid' else FOUR_FEATURES\n    out = (f.rename(columns={'proxy_masked': 'debt_raw'})[output_columns]")
    source = once(source, "'output_columns': list(OUTPUT_COLUMNS),\n        'model_features': list(MODEL_FEATURES),", "'output_columns': list(output_columns),\n        'model_features': list(model_features),")
    source = once(source, "    return out, record\n\n\ndef same_table", "    if debt_source == 'hybrid':\n        record['definition'] += '; debt_change=distinct available fiscal report delta (raw); rate_change=base-month rate delta pp (first month missing); policy_change_x_debt=policy_change*debt; rate_change_x_debt=rate_change*debt'\n        record['code_sha256'].update({p: sha256(CODE_ROOT / p) for p in ('src/h2/experiments/build_h2_debt_change.py', 'src/h2/preprocessing/build_bok_rate_history.py')})\n    return out, record\n\n\ndef same_table")
    source = once(source, "    if replace and (debt_source != 'hybrid' or not execute):", "    if execute and debt_source != 'hybrid':\n        raise ValueError('legacy_lag_6m_is_read_only; official h2.parquet requires hybrid eight features')\n    if replace and (debt_source != 'hybrid' or not execute):")
    source = once(source, "    target = root / config['PATHS']['h2']\n    if target.exists():", "    target = root / config['PATHS']['h2']\n    if debt_source == 'hybrid':\n        report = check_eight_archive(root, out)\n        record['archive_8feature_comparison'] = report\n        if execute and not report['match']:\n            print(json.dumps(report, ensure_ascii=False, indent=2))\n            raise ValueError('h2_eight_archive_mismatch; replacement blocked')\n    if target.exists():")
    start = "        if set(existing.columns) == LEGACY_OUTPUT_COLUMNS:\n"
    end = "        match = same_table(existing, out)\n"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Legacy migration block differs; no changes made')
    lo, hi = source.index(start), source.index(end)
    if lo >= hi:
        raise ValueError('Unexpected migration block order')
    source = source[:lo] + '''        if debt_source == 'hybrid':
            if set(existing.columns) not in (set(FOUR_OUTPUT_COLUMNS), set(OUTPUT_COLUMNS)):
                raise ValueError('h2_existing_unexpected_schema; replacement blocked')
            retained = compare_feature_tables(out, existing, FOUR_OUTPUT_COLUMNS)
            record['retained_4feature_comparison'] = retained
            record['old_sha256'] = sha256(target)
            if execute and not retained['match']:
                print(json.dumps(retained, ensure_ascii=False, indent=2))
                raise ValueError('h2_retained_four_features_differ; replacement blocked')
''' + source[hi:]
    source = once(source, "        elif debt_source == 'hybrid' and not execute:\n            record['status'] = 'DIFFERS_DRY_RUN'", "        elif not execute:\n            record['status'] = 'DIFFERS_DRY_RUN'")
    # A dry run does not write a parquet or a manifest, including when checks fail.
    # Keep original ensure_h2 backup and atomic parquet replacement implementation.
    compile(source, 'src/features/h2.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'src/features/h2.py'
    archive = root / 'data/processed/archive/h2_8features.parquet'
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Historical eight-feature archive is missing or its SHA256 differs. No source changed.')
    raw = path.read_bytes()
    before = raw.decode('utf-8-sig').replace('\r\n', '\n')
    after = patch(before)
    print(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(path), tofile='proposed h2.py')))
    if not args.apply:
        print('PREVIEW ONLY: no source or data changed. Use --apply to install.')
        return
    # Local backup outside the source tree; do not overwrite any previous backup.
    digest = hashlib.sha256(raw).hexdigest()
    backup = root / 'outputs/h2_code_cleanup/source_backups' / f'h2_{digest}.py.bak'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != raw:
            raise ValueError('Existing source backup mismatch')
    else:
        with backup.open('xb') as f:
            f.write(raw)
    if path.read_bytes() != raw:
        raise RuntimeError('Source changed during preview; aborting')
    # Preserve original newline convention and UTF-8 BOM, if present.
    encoded = after.replace('\n', '\r\n').encode('utf-8') if b'\r\n' in raw else after.encode('utf-8')
    if raw.startswith(b'\xef\xbb\xbf'):
        encoded = b'\xef\xbb\xbf' + encoded
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as f:
        tmp = Path(f.name)
        f.write(encoded)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print('SOURCE UPDATED:', path)
    print('ORIGINAL SOURCE BACKUP:', backup)
    print('No parquet, config or existing experiment files changed.')


if __name__ == '__main__':
    main()
