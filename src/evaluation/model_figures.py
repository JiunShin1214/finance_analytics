"""model_test 발표용 그림(보고용). model_summary.load_runs 결과와 각 실행 폴더의 curves.parquet만 읽는다. 재계산 없음.

그림 한 장 = 모델 하나·패널 하나, 그림 아래 설명 글 없음(설명은 발표에서 말로, 2026-11-15). 모델 간 비교는 표(model_summary)로 한다.
  ① learning_curve_<모델>.png  1등 설정, 한 Fold: 트리 수별 학습 IC·내부 검증 IC와 고른 트리 수
  ② depth_<모델>.png           순위 정답 모델: 깊이 1~4의 Fold 평균 학습 IC와 바깥 검증 IC
  ④ gain_<모델>.png            (기본 생성 안 함) 1등 설정의 Fold별 특징 중요도 — 발표에서 뺌(2026-11-15)
색: 학습=파랑, 검증=주황 / Size 파랑, B/M 주황, Mom 초록, σ 노랑, H1 지수 분홍(검증 스크립트 통과, 대비 경고는 범례·csv로 보완).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import model_summary as ms  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
TRAIN, VALID = '#2a78d6', '#eb6834'
FEATURES = {'size': ('Size', '#2a78d6'), 'bm': ('B/M', '#eb6834'), 'mom': ('Mom', '#1baf7a'),
            'h1_index': ('H1 지수', '#e87ba4'), 'sigma': ('σ', '#eda100')}
MODEL_LABELS = {'xgb_reg': 'XGBoost 회귀', 'lgb_reg': 'LightGBM 회귀', 'xgb_pairwise': 'XGBoost pairwise',
                'xgb_rankreg': 'XGBoost 순위 회귀', 'lgb_rankreg': 'LightGBM 순위 회귀'}
TARGET_LABELS = {'monthly_zscore': '월별 표준화', 'winsor_1_99': '윈저', 'percentile': '백분위', 'decile': '10분위'}
LOSS_LABELS = {'huber_sigma135': 'Huber②', 'mse': 'MSE', 'pairwise': 'pairwise'}
RANK_MODELS = ('xgb_pairwise', 'xgb_rankreg', 'lgb_rankreg')

def _korean_font() -> str:
    """설치된 한글 글꼴(Windows 맑은 고딕, Mac 애플 SD 고딕, Linux·Colab 나눔고딕 순). 없으면 기본 글꼴(한글이 깨질 수 있음)."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ('Malgun Gothic', 'Apple SD Gothic Neo', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR')
                 if n in names), 'DejaVu Sans')


plt.rcParams.update({'font.family': _korean_font(), 'axes.unicode_minus': False, 'font.size': 10,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'legend.frameon': False, 'lines.linewidth': 2})


def setting_label(row) -> str:
    depth = row['hparams'].split('_')[0].lstrip('d')
    return (f"{TARGET_LABELS.get(row['target'], row['target'])} · {LOSS_LABELS.get(row['loss'], row['loss'])} · 깊이 {depth} · "
            f"{ms.SET_LABELS.get(row['input_set'], row['input_set'])}")


def _note(ax, text, y=-0.15):
    ax.annotate(text, xy=(0, y), xycoords='axes fraction', fontsize=8.5, color=MUTED, ha='left', va='top')


def _match(frame: pd.DataFrame, row) -> pd.Series:
    """같은 설정의 행. 같은 설정이 두 폴더에 있을 수 있어 출처 폴더(run)가 있으면 그것도 맞춘다(곡선 파일엔 run 칸 없음)."""
    out = (frame.model.eq(row['model']) & frame.target.eq(row['target']) & frame.loss.eq(row['loss'])
           & frame.hparams.eq(row['hparams']) & frame.input_set.eq(row['input_set']))
    return out & frame.run.eq(row['run']) if 'run' in frame and 'run' in row else out


def _boosting_best(loaded: dict) -> pd.DataFrame:
    best = ms.best_per_model(loaded)
    return best[best.model.isin(MODEL_LABELS)].reset_index(drop=True)


def learning_curve(loaded: dict, row, out_dir: Path, *, fold: int = 16) -> Path:
    """① 1등 설정·한 Fold: 고른 λ의 트리별 내부 검증 IC와 10트리마다 학습 IC."""
    ft = loaded['fold_table']
    sel = ft[_match(ft, row) & ft.fold.eq(fold)].iloc[0]
    curves = ms.read_parquet(loaded['folders'][sel['run']], 'curves.parquet')
    c = curves[_match(curves, row) & curves.part.eq('fold') & curves.fold.eq(fold)]
    c = c[c.h3_label.eq(sel['h3_label'])] if isinstance(sel['h3_label'], str) else c[c.h3_label.isna()]
    c = c.iloc[0]
    inner = c.filter(like='inner_').to_numpy(float)
    train = c.filter(like='train_').to_numpy(float)
    every = len(inner) // len(train)
    x_in, x_tr = np.arange(1, len(inner) + 1), np.arange(every, len(inner) + 1, every)
    trees = int(sel['trees'])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x_tr, train, color=TRAIN, label='학습 IC (10트리마다)')
    ax.plot(x_in, inner, color=VALID, label='내부 검증 IC')
    ax.axvline(trees, color=MUTED, linewidth=1, linestyle='--')
    ax.plot([trees], [inner[trees - 1]], 'o', color=VALID, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f'고른 트리 {trees}개\n내부 검증 IC {inner[trees - 1]:.3f}', xy=(trees, inner[trees - 1]),
                xytext=(10, 22) if trees < len(inner) / 2 else (-10, 22), textcoords='offset points', fontsize=9,
                color=INK2, ha='left' if trees < len(inner) / 2 else 'right')
    ax.annotate('학습 IC', xy=(x_tr[-1], train[-1]), xytext=(4, 0), textcoords='offset points', fontsize=9, color=INK2, va='center')
    ax.annotate('내부 검증 IC', xy=(x_in[-1], inner[-1]), xytext=(4, 0), textcoords='offset points', fontsize=9, color=INK2,
                va='center')
    ax.set_xlim(-20, len(inner) * 1.1)       # 트리 수가 아주 적게 골라져도 점이 축에 가리지 않게
    ax.set_title(f"{MODEL_LABELS[row['model']]}: 트리 수별 학습·내부 검증 IC")
    ax.set_xlabel('트리 수')
    ax.set_ylabel('월별 순위 IC 평균')
    ax.legend(loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize=9)   # 두 곡선 사이 빈 공간
    fig.tight_layout()
    path = out_dir / f"learning_curve_{row['model']}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def depth_chart(loaded: dict, model: str, out_dir: Path, *, experiment: str = 'base') -> Path:
    """② 개별형 1등 목표·손실에서 깊이 1~4의 Fold 평균 학습 IC와 바깥 검증 IC(기본 실험만)."""
    p = loaded['performance']
    p = p[p.scope.eq('full') & p.model.eq(model) & p.input_set.eq('ind:size+bm+mom+sigma') & p.experiment.eq(experiment)]
    top = p.loc[p.mean_ic.idxmax()]
    p = p[p.target.eq(top.target) & p.loss.eq(top.loss)].copy()
    p['depth'] = p.hparams.str.extract(r'^d(\d+)_')[0].astype(int)
    p = p.sort_values('depth')
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(p.depth, p.train_ic_fold_mean, color=TRAIN, marker='o', markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label='학습 IC (Fold 평균)')
    ax.plot(p.depth, p.mean_ic, color=VALID, marker='o', markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label='바깥 검증 IC (192개월)')
    best = p.loc[p.mean_ic.idxmax()]
    ax.annotate(f'최고 깊이 {best.depth}: {best.mean_ic:.4f}', xy=(best.depth, best.mean_ic), xytext=(0, 14),
                textcoords='offset points', fontsize=9, color=INK2, ha='center')
    last = p.iloc[-1]
    ax.annotate(f'차이 {last.train_minus_valid:.3f}', xy=(last.depth, (last.train_ic_fold_mean + last.mean_ic) / 2),
                xytext=(8, 0), textcoords='offset points', fontsize=9, color=INK2, va='center')
    ax.set_xticks(p.depth, [f'{d}\n(후속 튜닝)' if d == 4 else str(d) for d in p.depth])
    ax.set_xlim(0.6, 4.6)
    ax.set_title(f'{MODEL_LABELS[model]}: 나무 깊이별 학습·검증 IC')
    ax.set_xlabel('나무 깊이')
    ax.set_ylabel('월별 순위 IC 평균')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    path = out_dir / f'depth_{model}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def gain_chart(loaded: dict, row, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    """④ 1등 설정의 검증 연도별 특징 중요도(gain 비중, 합 1). 막대 사이 2px 간격."""
    ft = loaded['fold_table']
    sel = ft[_match(ft, row)].sort_values('valid_year')
    cols = [c for c in ('gain_size', 'gain_bm', 'gain_mom', 'gain_h1_index', 'gain_sigma') if c in sel and sel[c].notna().any()]
    data = sel.set_index('valid_year')[cols].fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = np.zeros(len(data))
    for col in cols:
        label, color = FEATURES[col.removeprefix('gain_')]
        ax.bar(data.index, data[col], bottom=bottom, width=0.75, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        bottom += data[col].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_xticks(data.index, [str(y) for y in data.index], rotation=0, fontsize=8.5)
    ax.grid(axis='x', visible=False)
    ax.set_title(f"{MODEL_LABELS[row['model']]}: 검증 연도별 특징 중요도")
    ax.set_xlabel('바깥 검증 연도(Fold)')
    ax.set_ylabel('gain 비중')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(cols), fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    path = out_dir / f"gain_{row['model']}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, data.reset_index().assign(model=row['model'])


def make_figures(loaded: dict, out_dir, *, fold: int = 16) -> list[str]:
    """①·② (out_dir/figures/에 png). ④ 중요도 그림은 메시지가 불분명해 발표에서 뺌(2026-11-15), gain_chart는 필요할 때만 직접 호출."""
    out = Path(out_dir) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    best = _boosting_best(loaded)
    for _, row in best.iterrows():
        paths.append(learning_curve(loaded, row, out, fold=fold))
    for model in RANK_MODELS:
        if model in set(best.model):
            paths.append(depth_chart(loaded, model, out))
    return [str(p) for p in paths]


def tuning_trees_chart(summary_csv, model: str, out_dir, *, l2: int = 0, yearly_ranges: dict | None = None) -> Path:
    """튜닝 요약 표(tune_*_summary.csv)에서 한 모델의 트리 수(고정 500~2000)별 바깥 검증 IC, 깊이 3·4 두 선.
    해마다 선택은 Fold마다 트리 수가 달라 x 위치 없는 가로 점선(2026-11-23 발표). yearly_ranges = {깊이: (최소, 최대)} 고른 트리 수."""
    t = pd.read_csv(summary_csv)
    t = t[t['모델'].eq(model) & t['L2'].eq(l2)]
    colors = {3: TRAIN, 4: VALID}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for d, g in t.groupby('깊이'):
        fixed = g[g['트리'].str.endswith('고정')].assign(n=lambda f: f['트리'].str.split().str[0].astype(int)).sort_values('n')
        ax.plot(fixed.n, fixed['평균 IC'], color=colors[d], marker='o', markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=f'깊이 {d} (트리 고정)')
        y = g[g['트리'].eq('해마다')].iloc[0]
        rng = (yearly_ranges or {}).get(d)
        span = f', 고른 트리 {rng[0]:,}~{rng[1]:,}개' if rng else ''
        ax.hlines(y['평균 IC'], 250, 2000, color=colors[d], linewidth=1.5, linestyle=(0, (4, 3)),
                  label=f'깊이 {d} (해마다 선택{span})')
        ax.annotate(f"해마다 선택 {y['평균 IC']:.4f}", xy=(2000, y['평균 IC']), xytext=(8, 0), textcoords='offset points',
                    fontsize=8.5, color=INK2, va='center')
        ax.annotate(f'깊이 {d}', xy=(fixed.n.iloc[-1], fixed['평균 IC'].iloc[-1]), xytext=(8, 0), textcoords='offset points',
                    fontsize=9, color=INK2, va='center')
    best = t[t['트리'].str.endswith('고정')].loc[lambda f: f['평균 IC'].idxmax()]
    ax.annotate(f"최고: 깊이 {best['깊이']}, {best['트리']} {best['평균 IC']:.4f}", xy=(int(best['트리'].split()[0]), best['평균 IC']),
                xytext=(8, 8), textcoords='offset points', fontsize=9, color=INK2, va='bottom')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)
    ax.set_xticks([500, 1000, 1500, 2000])
    ax.set_xlim(250, 2400)
    ax.set_title(f'{MODEL_LABELS[model]}: 트리 수·깊이별 검증 IC')
    ax.set_xlabel('트리 수')
    ax.set_ylabel('월별 순위 IC 평균 (192개월)')
    ax.legend(loc='lower left', fontsize=8.5)
    fig.tight_layout()
    path = Path(out_dir) / f'tuning_trees_depth_{model}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
