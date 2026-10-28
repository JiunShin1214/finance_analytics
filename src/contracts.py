# 데이터 처리 단계 사이에서 사용할 표준 열 이름과 유일키를 정의한다.
# 패널·공시·특징·예측·평가 표의 형식을 맞추는 공통 기준으로 사용한다.

from __future__ import annotations

from typing import Any, Mapping

Config = Mapping[str, Any]

PANEL_COLUMNS = ("stock_id", "month", "return", "market_cap", "eligible")
SECURITY_MAP_COLUMNS = ("stock_id", "corp_code", "valid_from", "valid_to", "evidence")
DART_COLUMNS = ("corp_code", "fiscal_period", "published_at", "receipt_id", "equity", "statement_type", "account", "unit", "revision")
DOCUMENT_COLUMNS = ("document_id", "source", "published_at", "raw_path", "collection_status")
NEWS_METADATA_COLUMNS = ("title", "summary", "publication_date", "url", "collection_method")
POLICY_TEXT_COLUMNS = (*DOCUMENT_COLUMNS, "meeting_date", "release_date_source", "text", "extraction_status")
NEWS_ARTICLE_COLUMNS = (*DOCUMENT_COLUMNS, *NEWS_METADATA_COLUMNS, "extraction_status")
FEATURE_KEY = ("stock_id", "base_month")
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
QUALITY_COLUMNS = ("stage", "check", "status", "input_rows", "output_rows", "affected_scope", "reason", "correction_history", "input_id", "output_id")
