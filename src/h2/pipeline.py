"""Rebuild and verify the existing H2 Hybrid Train analysis in a fresh directory.

No Test prediction or network acquisition is performed here. Original data and
saved results remain read-only references. Hybrid is not fully verified PIT.
"""
from pathlib import Path
from datetime import datetime
import argparse
import contextlib
import json
import subprocess
import sys
import time
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, write_json, sha256


def fresh_output(value):
    out=Path(value)
    out=(out if out.is_absolute() else ROOT/out).resolve()
    # Do not allow an output that is an input directory, or a parent of inputs.
    if not out.is_relative_to((ROOT/'results').resolve()) or out==ROOT/'results':
        raise ValueError('Choose a fresh subdirectory under results/')
    if out.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {out}')
    out.mkdir(parents=True)
    return out


def compare_saved(work, full_disclosures):
    data=work/'data'
    pairs=[]
    for relative in ['policy/units.csv','policy/scores.csv','policy/monthly.csv','dataset.csv']:
        pairs.append((PROCESSED/relative,data/relative))
    for lag in [4,6,9]:
        for name in [f'h2_debt_lag_{lag}m_2001_2021.csv',f'debt_lag_{lag}m_monthly_stats.csv',
                     f'h2_debt_lag_{lag}m_fiscal_screened_2001_2021.csv']:
            pairs.append((PROCESSED/'debt'/name,data/'debt'/name))
    for folder in ['debt_disclosure','debt_hybrid','disclosures/reconciliation']:
        for ref in sorted((PROCESSED/folder).glob('*.csv')):
            pairs.append((ref,data/folder/ref.name))
    if full_disclosures:
        for ref in sorted((PROCESSED/'disclosures/bulk').glob('*.csv')):
            pairs.append((ref,data/'disclosures/bulk'/ref.name))
    for ref in sorted((ROOT/'results/validation/dart_versions').glob('*.csv')):
        pairs.append((ref,work/'dart_versions'/ref.name))
    for old,new in [('rankic_hybrid_comparison','rankic'),('h2_interaction_check','interaction')]:
        for ref in sorted((ROOT/'results'/old).glob('*.csv')):
            pairs.append((ref,work/new/ref.name))
    comparisons=[]
    for ref,new in pairs:
        item={'reference':str(ref.relative_to(ROOT)),'generated':str(new.relative_to(ROOT)),'passed':False}
        if ref.is_file() and new.is_file():
            a=pd.read_csv(ref,low_memory=False);b=pd.read_csv(new,low_memory=False)
            try:
                pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
                item['passed']=True
            except AssertionError as exc:
                item['difference']=str(exc)[:1200]
                # Current metadata uses a more explicit annual-title label.
                # Accept only this exact transition; all other columns still
                # have to match. Preserve the strict mismatch in the report.
                if ref.name=='subsequent_filings_review_queue.csv' and 'period_status' in a and 'period_status' in b:
                    if a.period_status.eq('STANDARD_MONTH_LABEL_NOT_VERIFIED').all() and b.period_status.eq('DEC_ANNUAL_TITLE_CANDIDATE').all():
                        try:
                            pd.testing.assert_frame_equal(a.drop(columns='period_status'),b.drop(columns='period_status'),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
                        except AssertionError:
                            pass
                        else:
                            item.update(passed=True,metadata_label_change=True,
                                        note='Annual title label changed; every other column matches. This does not approve fiscal-screen changes.')
            item.update(rows=len(b),reference_sha256=sha256(ref),generated_sha256=sha256(new),
                        byte_equal=sha256(ref)==sha256(new))
        else:item['difference']='Missing reference or generated file'
        comparisons.append(item)
    hybrid=json.loads((data/'debt_hybrid/summary.json').read_text(encoding='utf-8'))
    expected={'total_rows':175022,'hybrid_rows':158799,'dart_selected_rows':15197,
              'eligible_dart_events':2146,'unresolved_followup_stock_months':3983,'date_violations':0}
    checks={key:hybrid.get(key)==value for key,value in expected.items()}
    config=json.loads((work/'rankic/run_config.json').read_text(encoding='utf-8'))
    checks['common_rows']=config['four_way_common_rows']==154215
    summary=pd.read_csv(work/'rankic/rank_ic_summary.csv')
    ols=summary.loc[summary['sample'].eq('common') & summary.algorithm.eq('OLS') & summary.feature_set.eq('policy_debt_size')]
    expected_ic={'legacy':.030204,'6m':.026597,'9m':.025081,'hybrid':.026647}
    checks['ols_ic']=len(ols)==4 and all(round(r.mean_ic,6)==expected_ic[r.scenario] and r.n_months==108 for r in ols.itertuples())
    current=pd.read_csv(work/'rankic/monthly_rank_ic.csv')
    baseline=pd.read_csv(ROOT/'results/rankic_hybrid_comparison/monthly_rank_ic.csv')
    keys=['sample','scenario','fold','target_month','model']
    merged=current.merge(baseline,on=keys,how='outer',suffixes=('_new','_old'),indicator=True,validate='1:1')
    maximum=float((merged.rank_ic_new-merged.rank_ic_old).abs().max())
    checks['monthly_keys']=bool(merged['_merge'].eq('both').all())
    checks['monthly_ic']=maximum<=1e-12
    result={'passed':bool(comparisons and all(x['passed'] for x in comparisons) and all(checks.values())),
            'strict_all_tables_equal':all(x['passed'] and not x.get('metadata_label_change') for x in comparisons),
            'comparisons':comparisons,'checks':checks,'monthly_max_abs_ic_difference':maximum,
            'hybrid_summary':hybrid,'fully_pit_verified':False}
    write_json(result,work/'regression_results.json')
    if not result['passed']:raise AssertionError(f'Reproduction differs: {work / "regression_results.json"}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('results/reproduction')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--full-disclosures',action='store_true',help='Re-extract all original ZIPs; otherwise use preserved extraction CSVs')
    parser.add_argument('--validate-only',action='store_true',help='Compatibility flag; this pipeline always performs Train reproduction only')
    args=parser.parse_args()
    from src.h2.features.prepare_policy import prepare
    from src.h2.features.score_policy import score
    from src.h2.preprocessing.build_dataset import build
    from src.h2.preprocessing.prepare_debt import prepare as debt
    from src.h2.preprocessing.prepare_disclosure_debt import build_monthly_panel
    from src.h2.evaluation.validate_disclosures import metadata,extract_originals,reconcile,build_candidate
    from src.h2.evaluation.check_dart_revisions import main as revisions
    work=fresh_output(args.output_root);data=work/'data';logs=work/'logs';logs.mkdir()
    steps=[]
    def step(name,fn):
        print(f'[{name}]',flush=True);start=time.monotonic()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):fn()
        steps.append({'name':name,'status':'PASS','seconds':round(time.monotonic()-start,2)})
    def module(name,*options):
        subprocess.run([sys.executable,'-B','-m',name,*map(str,options)],cwd=ROOT,stdout=sys.stdout,stderr=sys.stderr,check=True)
    status={'fully_pit_verified':False,'test_prediction_implemented':False,'steps':steps}
    try:
        step('prepare_policy',lambda:prepare(output=data/'policy'))
        step('score_policy',lambda:score(data/'policy'))
        step('build_dataset',lambda:build(policy_path=data/'policy/monthly.csv',output=data/'dataset.csv'))
        step('disclosure_metadata',lambda:metadata(data/'disclosures'))
        # Reproduce the approved saved experiment with its recorded input screen.
        # The current metadata parser also recognizes March/June/September annual
        # fiscal periods. Its broader screen is preserved as a review candidate;
        # silently substituting it would change the saved experiment's universe.
        reference=ROOT/'data/reference/nonstandard_period_filings_reference.csv'
        current_reference=data/'disclosures/nonstandard_period_filings_reference.csv'
        def issuers(path):
            d=pd.read_csv(path,dtype={'stock_code':'string'})
            return set(d.loc[d.report_type.eq('사업보고서') & d.fiscal_month_label.ne(12) & d.fiscal_year_label.between(2000,2025),'stock_code'])
        saved,current=issuers(reference),issuers(current_reference)
        write_json({'training_screen':'RECORDED_EXPERIMENT_REFERENCE','reference':str(reference.relative_to(ROOT)),
                    'reference_sha256':sha256(reference),'recorded_issuer_count':len(saved),
                    'current_metadata_issuer_count':len(current),'additional_issuers':sorted(current-saved),
                    'removed_issuers':sorted(saved-current),'requires_review':saved!=current,
                    'note':'Current metadata rule retained. New screen is not silently promoted into the historical experiment.'},work/'fiscal_screen_review.json')
        if saved!=current:print(f'[fiscal_screen_review] recorded={len(saved)}, current={len(current)}; reproducing recorded screen',flush=True)
        step('prepare_debt',lambda:debt(data/'dataset.csv',reference,data/'debt'))
        extracted=PROCESSED/'disclosures/bulk'
        if args.full_disclosures:
            extracted=data/'disclosures/bulk'
            step('disclosure_originals',lambda:extract_originals(extracted))
        step('disclosure_reconciliation',lambda:reconcile(data/'disclosures/reconciliation',extracted,data/'dataset.csv'))
        step('disclosure_candidate',lambda:build_candidate(data/'disclosures/reconciliation',data/'dataset.csv'))
        step('prepare_disclosure_debt',lambda:build_monthly_panel(data/'dataset.csv',data/'disclosures/reconciliation/dataguide_dart_event_reconciliation.csv',data/'disclosures/dart_disclosure_history.csv',data/'debt_disclosure','2001-01','2021-12'))
        step('audit_dart_versions',lambda:module('src.h2.evaluation.audit_dart_versions',
             '--history',data/'disclosures/dart_disclosure_history.csv','--events',data/'debt_disclosure/accepted_filing_events.csv',
             '--monthly',data/'debt_disclosure/h2_debt_disclosure_asof_candidate.csv','--output',work/'dart_versions'))
        step('check_dart_revisions',lambda:revisions(work/'dart_versions',data/'disclosures/dart_disclosure_history.csv',
             data/'debt_disclosure/accepted_filing_events.csv',ROOT/'results/validation/dart_versions/followup_originals/originals'))
        step('prepare_debt_hybrid',lambda:module('src.h2.preprocessing.prepare_debt_hybrid',
             '--proxy',data/'debt/h2_debt_lag_6m_fiscal_screened_2001_2021.csv','--audit',work/'dart_versions/dart_event_version_audit.csv','--output',data/'debt_hybrid'))
        hybrid=data/'debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv'
        step('evaluate_rankic',lambda:module('src.h2.evaluation.evaluate_rankic','--panel',data/'dataset.csv',
             '--lag-dir',data/'debt','--hybrid',hybrid,'--out',work/'rankic'))
        step('evaluate_h2_interaction',lambda:module('src.h2.evaluation.evaluate_h2_interaction','--panel',data/'dataset.csv',
             '--lag-dir',data/'debt','--hybrid',hybrid,'--previous',work/'rankic','--out',work/'interaction'))
        step('regression_comparison',lambda:compare_saved(work,args.full_disclosures))
        status['status']='PASS'
        print(f'Hybrid Train reproduction PASS: {work}',flush=True)
    except Exception as exc:
        status.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:write_json(status,work/'pipeline_status.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
