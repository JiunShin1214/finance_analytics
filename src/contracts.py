# 데이터 처리 단계 사이에서 사용할 표준 열 이름과 유일키를 정의한다.
# 패널·공시·특징·예측·평가 표의 형식을 맞추는 공통 기준으로 사용한다.

from __future__ import annotations

from typing import Any, Mapping

Config = Mapping[str, Any]

# 공통 공시 입력: 실행 이력·재개 상태·선정 결과를 포함하지 않는다.
DART_FILING_INPUT_COLUMNS = (
    'corp_code','corp_name','stock_code','receipt_id','published_at','report_name',
    'report_kind','report_period_label','revision','listed_in_public_index',
    'disclosure_format','source_path','source_sha256',
)
DART_FACT_INPUT_COLUMNS = (
    'corp_code','receipt_id','concept','statement_type','equity','amount_raw','unit',
    'fiscal_period','fiscal_period_basis','source_kind','section','column_basis',
    'account_name','member','location','identity_ok','raw_path','sha256','components_match',
)

# 원천 변수 -> 관측 패널 필드. 표시용 레이블과 구분한다.
WIDE_VARIABLE_COLUMNS = {
    "adjusted_monthly_return": "return",
    "raw_month_end_market_cap": "market_cap",
    "raw_month_end_price": "price",
    "raw_month_end_shares_outstanding": "shares_outstanding",
    "company_name": "company_name",
    "isin": "isin",
}
NUMERIC_PANEL_UNITS = {
    "return": "decimal", "market_cap": "KRW", "price": "KRW",
    "shares_outstanding": "shares",
}
DEVELOPMENT_MONTHS = {
    "warmup_2000": ("2000-01", "2000-12"),
    "train_2001_2021": ("2001-01", "2021-12"),
}
PANEL_COLUMNS = ("stock_id", "month", "return", "market_cap", "eligible")
# eligible는 원자료에서 확인 불가: nullable boolean 결측, 적격성 인증 아님.
WIDE_PANEL_COLUMNS = (*PANEL_COLUMNS, "price", "shares_outstanding", "company_name", "isin")
SECURITY_MAP_COLUMNS = ("stock_id", "corp_code", "valid_from", "valid_to", "evidence")
DART_COLUMNS = ("corp_code", "fiscal_period", "published_at", "receipt_id", "equity", "statement_type", "account", "unit", "revision")
# 종목×월 적격성(그 달 마지막 거래일에 KOSPI 보통주로 상장, src.preprocessing.listing). data/interim/eligibility.parquet.
# panel의 eligible은 이 표의 eligible을 그대로 채운다. 판정 근거·상장일·폐지일은 이 표에만 둔다.
ELIGIBILITY_COLUMNS = ("stock_id", "month", "eligible", "eligibility_basis", "listing_date", "delisting_date")
# 월별 마지막 거래일(marcap 거래 기록 기준). data/interim/trading_days.parquet.
TRADING_DAY_COLUMNS = ("month", "last_trading_day")
# 최종 자기자본 as-of(src.preprocessing.equity_asof, data/interim/asof.parquet)의 실제 열. 키 이름은 stock_code를 유지한다.
EQUITY_ASOF_COLUMNS = (
    "stock_code", "base_month", "equity", "tier", "statement_month", "statement_date", "source", "rcept_no", "rcept_dt",
    "available_month", "expiry_date", "equity_age_months", "possibly_corrected", "market_cap_present", "usable", "reason",
)
# DART 원문만으로 만드는 as-of 중간표(dart_extract.join_equity_asof, equity_asof의 DART 전용 보충에 쓰임).
# 최종 asof.parquet의 열은 EQUITY_ASOF_COLUMNS다. 결측 행도 유지하고 사유를 asof_status/exclusion_reason에 남긴다.
ASOF_COLUMNS = (
    "stock_id", "base_month", "corp_code", "mapping_status", "mapping_basis",
    "equity", "equity_tier", "statement_type", "account_name", "source_kind", "section", "column_basis",
    "receipt_id", "published_at", "availability_month", "report_kind", "report_period_label",
    "fiscal_period_end", "fiscal_period_basis", "revision", "report_name", "raw_path", "sha256", "identity_ok", "xbrl_document_agree",
    "owners_resolution", "override_id",
    "candidates_n", "fallback", "skipped_receipt_ids", "skipped_reasons", "same_day_order_unverified",
    "equity_age_months", "expiry_date", "scope_changed", "history_incomplete",
    "asof_status", "usable", "exclusion_reason",
)
DOCUMENT_COLUMNS = ("document_id", "source", "published_at", "raw_path", "collection_status")
NEWS_METADATA_COLUMNS = ("title", "summary", "publication_date", "url", "collection_method")
POLICY_TEXT_COLUMNS = (*DOCUMENT_COLUMNS, "meeting_date", "release_date_source", "text", "extraction_status")
NEWS_ARTICLE_COLUMNS = (*DOCUMENT_COLUMNS, *NEWS_METADATA_COLUMNS, "extraction_status")
FEATURE_KEY = ("stock_id", "base_month")
H1_COMPONENT_COLUMNS = (*FEATURE_KEY, "eligible", "size", "bm", "mom", "mom_valid_months",
                        "size_available", "bm_available", "mom_available",
                        "size_exclusion_reason", "bm_exclusion_reason", "mom_exclusion_reason")
MOM_COLUMNS = (*FEATURE_KEY, "eligible", "mom", "mom_available", "mom_valid_months", "mom_exclusion_reason")
H1_COLUMNS = (*FEATURE_KEY, "eligible", "z_size", "z_bm", "z_mom", "h1", "standardization_n",
              "h1_available", "h1_exclusion_reason")
# H3 공유 표(data/processed/h3.parquet). sigma=sqrt(v)는 산출 가능할 때만, h3는 월별 Z(ddof=1) 원값(2026-11-08 원값 유지 결정).
H3_COLUMNS = (*FEATURE_KEY, "eligible", "sigma", "v", "initialized", "initialization_month", "recent_valid_count",
              "h3_available", "h3", "standardization_n", "h3_exclusion_reason")
TARGET_KEY = ("stock_id", "target_month")
FEATURE_COLUMNS = (*FEATURE_KEY, "target_month", "h1", "h3", "mom", "available", "feature_exclusion_reason")
SAMPLE_COLUMNS = (*FEATURE_KEY, "target_month", "h1", "h3", "y", "split", "sample_id", "target_exclusion_reason")
# 확장 열은 활성화된 가설에 대해서만 요구한다. 기준모형 표본에는 영향을 주지 않는다.
H2_VALUE_COLUMNS = ("policy_score", "sensitivity", "h2_interaction")
H2_STATUS_COLUMNS = ("h2_available", "h2_exclusion_reason")
H4_VALUE_COLUMNS = ("news_ratio", "h4_interaction")
H4_STATUS_COLUMNS = ("h4_available", "h4_exclusion_reason")
H2_FEATURE_COLUMNS = (*FEATURE_KEY, *H2_VALUE_COLUMNS, *H2_STATUS_COLUMNS)
H4_FEATURE_COLUMNS = (*FEATURE_KEY, "mom", *H4_VALUE_COLUMNS, *H4_STATUS_COLUMNS)
EXTENDED_FEATURE_COLUMNS = (
    *FEATURE_COLUMNS, *H2_VALUE_COLUMNS, *H2_STATUS_COLUMNS,
    *H4_VALUE_COLUMNS, *H4_STATUS_COLUMNS,
)
EXTENDED_SAMPLE_COLUMNS = (
    *SAMPLE_COLUMNS, "mom", *H2_VALUE_COLUMNS, *H2_STATUS_COLUMNS,
    *H4_VALUE_COLUMNS, *H4_STATUS_COLUMNS,
)
PREDICTION_KEY = ("model", "candidate", "fold", "stock_id", "target_month")
PREDICTION_COLUMNS = (*PREDICTION_KEY, "yhat", "y")
IC_COLUMNS = ("model", "candidate", "fold", "target_month", "n", "ic", "reason")
# 시계열 평균 검정 공통 열. 대응 차이 표에서는 mean/sample_sd를
# mean_delta_ic/delta_ic_sd로 바꾸고 holm_p/significant_holm_05를 추가한다.
H1_HAC_TEST_COLUMNS = (
    "valid_months", "mean", "sample_sd", "hac_se", "statistic", "p_two_sided",
    "ci95_low", "ci95_high", "lag", "df", "reference_distribution", "kernel",
    "small_sample_correction", "significant_raw_05", "usual_se", "usual_t",
    "usual_p_two_sided", "reason",
)
QUALITY_COLUMNS = ("stage", "check", "status", "input_rows", "output_rows", "affected_scope", "reason", "correction_history", "input_id", "output_id")

# 10주차 H1 진단: 기존 IC_COLUMNS를 유지하고 기준월·표본 종류를 덧붙인다.
H1_MONTHLY_IC_COLUMNS = (*IC_COLUMNS, "base_month", "sample_scope")
H1_PCA_SCORE_COLUMNS = (*FEATURE_KEY, "eligible", "candidate", "score",
                        "score_available", "score_exclusion_reason")
H1_CONSTRAINED_PCA_COLUMNS = (
    "method", "base_month", "target_month", "current_n", "fit_n", "fit_months",
    "fit_first_base_month", "fit_last_base_month", "available", "exclusion_reason",
    "projected_variance", "explained_variance_ratio", "ordinary_pc1_variance_ratio",
    "variance_ratio_delta", "optimal_solutions", "weight_neg_size", "weight_bm", "weight_mom",
    "center_neg_size", "center_bm", "center_mom",
)

# H1 주성분 진단은 설명분산만 저장하며 점수·선택·수익률 평가를 포함하지 않는다.
H1_PCA_VARIANCE_COLUMNS = (
    "method", "base_month", "target_month", "current_n", "fit_n", "fit_months",
    "fit_first_base_month", "fit_last_base_month", "component", "eigenvalue",
    "explained_variance_ratio", "cumulative_explained_variance_ratio", "available",
    "exclusion_reason", "tiny_negative_eigenvalue_clipped",
)

# 주성분 계산 가중치와 특징-성분 상관을 구분한다. fit/current는 추정/당월 집합.
H1_PCA_STRUCTURE_COLUMNS = (
    "weight_neg_size", "weight_bm", "weight_mom",
    "fit_corr_neg_size", "fit_corr_bm", "fit_corr_mom",
    "current_corr_neg_size", "current_corr_bm", "current_corr_mom",
    "fit_corr_reason", "current_corr_reason", "display_reference_base_month",
    "display_sign_flipped", "previous_axis_cosine", "nearest_eigenvalue_gap_ratio",
    "eigenvector_identifiable",
)
