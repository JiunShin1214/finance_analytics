"""H4 input preparation using current team APIs; model fitting is not implemented here.

The former adapter omitted main effects and ranked away monthly EPU variation.
Use prepare_for_model only after the current common code and inputs are present.
No claim of official result.zip validation or incremental H1 forecast performance.
"""
from pathlib import Path
import argparse
import inspect
import yaml
from .h4_export import export,extra_feature_specs


def prepare_for_model(root=Path.cwd()):
    root=Path(root).resolve()
    try:
        from src.evaluation import model_test as mt
    except ImportError as exc:
        raise RuntimeError('최신 팀 model_test와 공통 입력을 먼저 준비해야 합니다. 현재 모델 결합 미검증.') from exc
    if 'extra_features' not in inspect.signature(mt.prepare_inputs).parameters or 'transforms' not in inspect.signature(mt.add_extra_features).parameters:
        raise RuntimeError('transform=none을 지원하는 최신 팀 API가 필요합니다.')
    config=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'))
    required=['panel','eligibility','h1_components']
    missing=[config.get('PATHS',{}).get(n,n) for n in required if not config.get('PATHS',{}).get(n) or not (root/config['PATHS'][n]).is_file()]
    if 'MODEL_GRID' not in config or missing:
        raise RuntimeError(f'최신 설정/입력 부족: MODEL_GRID={"MODEL_GRID" in config}, missing={missing}')
    export(root)
    prepared=mt.prepare_inputs(root,config,extra_features=extra_feature_specs())
    return prepared,config

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='입력 준비만 수행; H1 대비 모형 적합/공식 결과 저장 미실행')
    parser.add_argument('--prepare-only',action='store_true',required=True)
    args=parser.parse_args()
    prepared,config=prepare_for_model()
    print('입력 준비 완료. 기준/확장모형의 입력 조합과 평가 조건을 확정한 뒤 공통 실행기로 평가해야 합니다.')
    print(prepared['checks'])
