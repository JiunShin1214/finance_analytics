"""Final Rule v2; unchanged rules and rebuilt monthly as-of mapping."""

def score(directory=None):
    from src.h2.data_io import assert_unique
    directory = Path(directory) if directory else PROCESSED/'policy'
    units = pd.read_csv(directory/'units.csv')
    units['document_id'] = units.document_id.astype('string').str.strip()
    values = units.text_unit.apply(classify_policy_text)
    for i, col in enumerate(['final_label','final_stance_value','final_rule_id']):
        units[col] = [v[i] for v in values]
    docs = build_document_scores(units)
    release = pd.read_csv(directory/'release_dates.csv')
    docs = docs.merge(release, on='document_id', how='left', validate='1:1')
    assert_unique(docs, ['document_id'])
    rows = []
    for month in pd.period_range('2001-01','2021-12',freq='M'):
        row = {'month':str(month)}
        for mode in ['strict','sensitivity']:
            dates = pd.to_datetime(docs[f'release_date_{mode}'],errors='coerce')
            # Preserve the historical next-day availability convention. A release
            # on the final calendar day belongs to the following feature month.
            available = docs.loc[docs.final_PolicyScore.notna() & dates.notna() & (dates<month.end_time.normalize())].copy()
            available['_available'] = dates.loc[available.index]
            available = available.sort_values(['_available','document_id'])
            chosen = available.iloc[-1] if len(available) else None
            row[f'final_rule_v2_{mode}'] = chosen.final_PolicyScore if chosen is not None else np.nan
            row[f'{mode}_source_document_id'] = chosen.document_id if chosen is not None else pd.NA
            row[f'{mode}_source_release_date'] = chosen['_available'].date().isoformat() if chosen is not None else pd.NA
        rows.append(row)
    write_csv(units,directory/'scored_units.csv')
    write_csv(docs,directory/'scores.csv')
    write_csv(pd.DataFrame(rows),directory/'monthly.csv')
    return directory/'monthly.csv'
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, VALIDATION, write_csv, write_json

def normalize_text(text):
    if pd.isna(text):
        return ''
    return re.sub('\\s+', ' ', str(text)).strip()

def compact_text(text):
    """
    PDF/HWP 추출 과정에서 발생한 띄어쓰기 차이를 줄이기 위해
    규칙 매칭용 텍스트에서는 공백을 제거한다.
    """
    return re.sub('\\s+', '', normalize_text(text))

def has_any(text, patterns):
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

MAIN_RATE_TERM = '(?:(?:한국은행)?기준금리|콜금리(?:\\([^)]{0,50}\\))?(?:목표)?)'

SUPPORT_RATE_TERM = '(?:유동성조절대출금리|금융중개지원대출금리|총액대출(?:한도)?금리)'

DOVISH_RATE_ACTION = [f'{MAIN_RATE_TERM}.{{0,100}}[0-9.]+%?.{{0,25}}(?:에서|→).{{0,25}}[0-9.]+%?.{{0,35}}(?:하향조정|인하)', f'{MAIN_RATE_TERM}.{{0,80}}(?:인하함|인하하여|인하하기로|인하하였|인하하였다|하향조정함|하향조정하여|하향조정하기로|하향조정하였)', f'{SUPPORT_RATE_TERM}.{{0,80}}(?:인하함|인하하여|인하하기로|인하하였|하향조정)']

HAWKISH_RATE_ACTION = [f'{MAIN_RATE_TERM}.{{0,100}}[0-9.]+%?.{{0,25}}(?:에서|→).{{0,25}}[0-9.]+%?.{{0,35}}(?:상향조정|인상)', f'{MAIN_RATE_TERM}.{{0,80}}(?:인상함|인상하여|인상하기로|인상하였|인상하였다|상향조정함|상향조정하여|상향조정하기로|상향조정하였)']

DOVISH_STANCE = ['통화정책.{0,20}완화기조.{0,20}(?:유지|지속|이어)', '금융완화기조.{0,20}(?:유지|지속|이어)', '완화기조.{0,20}(?:유지|지속|이어)', '통화정책.{0,25}완화적으로.{0,20}(?:운용|운영)', '통화정책의?완화정도.{0,20}(?:확대|강화)', '금융완화의?정도.{0,20}(?:확대|강화)', '유동성(?:을|를).{0,20}(?:신축적으로|충분히|원활히)?.{0,10}공급', '기업부문으로의?자금공급확대.{0,25}(?:노력|지원|도모)']

HAWKISH_STANCE = ['완화정도.{0,15}(?:점진적으로|적절히|단계적으로)?.{0,15}조정해나갈', '완화정도.{0,20}(?:축소|줄여|낮춰)', '금융완화의?정도.{0,20}(?:축소|줄어|낮아)', '완화기조.{0,20}(?:축소|정상화)', '통화정책.{0,20}(?:긴축|긴축적)', '유동성(?:을|를).{0,20}(?:억제|흡수|축소)', '과잉유동성.{0,30}(?:방지|억제)']

NEUTRAL_HOLD = ['(?:한국은행)?기준금리.{0,150}콜금리(?:\\([^)]{0,50}\\))?(?:목표)?.{0,100}(?:같은|동일한|동일).{0,50}(?:설정|결정)', '콜금리(?:\\([^)]{0,50}\\))?(?:목표)?.{0,100}(?:같은|동일한|동일).{0,100}(?:한국은행)?기준금리.{0,80}(?:설정|결정)', f'{MAIN_RATE_TERM}.{{0,100}}(?:현수준|현재수준).{{0,40}}(?:유지|동결)', f'{MAIN_RATE_TERM}.{{0,100}}현재의?.{{0,30}}[0-9.]+%?.{{0,30}}(?:유지|동결)', f'{MAIN_RATE_TERM}.{{0,70}}(?:유지하기로|동결하기로|그대로유지)']

NEUTRAL_MONITOR = ['완화정도.{0,25}(?:조정여부|추가조정여부).{0,25}(?:판단|점검|검토)', '완화정도의?(?:추가)?조정여부.{0,30}(?:신중히)?판단']

PAST_POLICY_EFFECT = [f'{MAIN_RATE_TERM}.{{0,25}}(?:인상|인하).{{0,25}}(?:이후|영향|효과)', '(?:금리인상|금리인하).{0,25}(?:이후|영향|효과)', '(?:콜금리목표인상|콜금리목표인하).{0,25}(?:이후|영향|효과)', '그동안의?.{0,20}(?:콜금리|기준금리).{0,20}(?:인하|인상).{0,20}(?:영향|효과)', '지난.{0,20}(?:콜금리|기준금리).{0,20}(?:인하|인상).{0,20}(?:이후|영향|효과)']

FOREIGN_POLICY = ['(?:미국|미연준|연준|FRB|FederalReserve).{0,40}(?:금리|통화정책|정상화)', '(?:ECB|유럽중앙은행|일본은행|BOJ).{0,40}(?:금리|통화정책)', '주요국중앙은행.{0,40}(?:통화정책|금리)', '주요국.{0,30}(?:완화적|긴축적)?.{0,15}통화정책']

DOMESTIC_POLICY_CUE = ['금융통화위원회', '한국은행', '기준금리', '콜금리', '통화정책.{0,30}(?:운용|운영|완화|긴축|조정)', '금융중개지원대출', '유동성조절대출']

def classify_policy_text(raw_text):
    t = compact_text(raw_text)
    if has_any(t, DOVISH_RATE_ACTION):
        return ('DOVISH', 1.0, 'D1_RATE_CUT')
    if has_any(t, HAWKISH_RATE_ACTION):
        return ('HAWKISH', -1.0, 'H1_RATE_HIKE')
    if has_any(t, HAWKISH_STANCE):
        return ('HAWKISH', -1.0, 'H2_REDUCE_ACCOMMODATION')
    if has_any(t, DOVISH_STANCE):
        return ('DOVISH', 1.0, 'D2_EASING_STANCE')
    if has_any(t, NEUTRAL_HOLD):
        return ('NEUTRAL', 0.0, 'N1_RATE_HOLD')
    if has_any(t, NEUTRAL_MONITOR):
        return ('NEUTRAL', 0.0, 'N2_MONITOR_ADJUSTMENT')
    if has_any(t, PAST_POLICY_EFFECT):
        return ('NON_POLICY', np.nan, 'NP1_PAST_POLICY_EFFECT')
    foreign = has_any(t, FOREIGN_POLICY)
    domestic = has_any(t, DOMESTIC_POLICY_CUE)
    if foreign and (not domestic):
        return ('NON_POLICY', np.nan, 'NP2_FOREIGN_POLICY')
    return ('NON_POLICY', np.nan, 'NP0_BACKGROUND')

def build_document_scores(units):
    rows = []
    for doc_id, g in units.groupby('document_id'):
        counts = g['final_label'].value_counts()
        dovish_n = int(counts.get('DOVISH', 0))
        hawkish_n = int(counts.get('HAWKISH', 0))
        neutral_n = int(counts.get('NEUTRAL', 0))
        nonpolicy_n = int(counts.get('NON_POLICY', 0))
        if dovish_n + hawkish_n > 0:
            score = (dovish_n - hawkish_n) / (dovish_n + hawkish_n)
        elif neutral_n > 0:
            score = 0.0
        else:
            score = np.nan
        rows.append({'document_id': str(doc_id).strip(), 'dovish_n': dovish_n, 'hawkish_n': hawkish_n, 'neutral_n': neutral_n, 'nonpolicy_n': nonpolicy_n, 'directional_n': dovish_n + hawkish_n, 'policy_related_n': dovish_n + hawkish_n + neutral_n, 'final_PolicyScore': score})
    return pd.DataFrame(rows)
