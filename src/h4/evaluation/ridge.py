"""Paired development-only H4 Ridge OOS comparison; no shared-code mutations."""

from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml
from src.evaluation import model_test as mt
from ..preprocessing.export import extra_feature_specs

BASE = "ind:size+bm+mom+sigma"
EXT = BASE + "+h4_epu+h4_interaction"
SETS = {
    BASE: ["size", "bm", "mom", "sigma"],
    EXT: ["size", "bm", "mom", "sigma", "h4_epu", "h4_interaction"],
}


def protected_hashes(root):
    paths = [
        *root.joinpath("config").rglob("*"),
        *root.joinpath("src/features").rglob("*.py"),
        root / "src/h4/preprocessing/export.py",
        root / "src/h4/evaluation/feature_rank.py",
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file() and "__pycache__" not in str(p)
    }


def prepare(root):
    config = yaml.safe_load(
        (root / "config/experiment.yaml").read_text(encoding="utf-8")
    )
    mt.check_no_test(config)
    if int(config["MODEL_GRID"]["last_validation_year"]) > 2021:
        raise PermissionError("final_test_validation_year_blocked")
    required = [
        config["PATHS"][name] for name in ("panel", "eligibility", "h1_components")
    ]
    required.append("data/processed/h4_epu.parquet")
    for relative in required:
        path = (root / relative).resolve()
        if not path.is_relative_to((root / "data").resolve()):
            raise PermissionError("input_must_be_under_data")
        if any(
            token in path.name.lower()
            for token in ("test", "2022", "2023", "2024", "2025")
        ):
            raise PermissionError("final_test_path_blocked")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing development input: {relative}. See src/h4/README.md."
            )
    specs = {k: v for k, v in extra_feature_specs().items() if k != "h4_mom"}
    # Use the existing H4 parquet; never regenerate/overwrite existing inputs or outputs.
    prepared = mt.prepare_inputs(root, config, extra_features=specs)
    d = prepared["data"]
    if not d.target_month.lt("2022-01").all():
        raise PermissionError("final_test_target_month_blocked")
    if not d.base_month.le("2021-11").all():
        raise PermissionError("final_test_base_month_blocked")
    if int(config["MODEL_GRID"]["last_validation_year"]) > 2021:
        raise PermissionError("final_test_validation_year_blocked")
    if int(prepared["checks"]["train_start_target_year"]) != 2001:
        raise ValueError("comparison_requires_training_from_2001")
    if d.loc[d.h4_epu_available].groupby("base_month").x_h4_epu.nunique().max() != 1:
        raise ValueError("epu_must_be_monthly_cross_section_constant")
    return prepared, config


def scale_fold(prepared, fold):
    """Fit all numeric-feature scales on inner-fit rows only, before missing=0 fill."""
    p = dict(prepared)
    d = prepared["data"].copy()
    fit = mt.fold_rows(prepared, fold)["fit"]
    columns = [
        "x_size",
        "x_bm",
        "x_mom",
        *["x_sigma_" + x for x in prepared["sigma_labels"]],
        "x_h4_epu",
        "x_h4_interaction",
    ]
    x = d.loc[fit, columns].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_training_features")
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    if (sd <= 0).any():
        raise ValueError("constant_training_feature")
    d[columns] = (d[columns] - mean) / sd
    # Check intercept + design full column rank for every sigma candidate.
    diagnostics = []
    for label in prepared["sigma_labels"]:
        cols = [
            "x_size",
            "x_bm",
            "x_mom",
            "x_sigma_" + label,
            "x_h4_epu",
            "x_h4_interaction",
        ]
        matrix = np.column_stack([np.ones(fit.sum()), d.loc[fit, cols].to_numpy(float)])
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank != matrix.shape[1]:
            raise ValueError("perfect_multicollinearity")
        diagnostics.append(
            dict(
                fold=fold["fold"],
                h3_label=label,
                design_rank=rank,
                design_columns=matrix.shape[1],
                condition_number=float(singular[0] / singular[-1]),
            )
        )
    p["data"] = d
    scales = pd.DataFrame(
        {
            "fold": fold["fold"],
            "column": columns,
            "fit_mean": mean,
            "fit_sd": sd,
            "fit_rows": int(fit.sum()),
            "fit_last_target_month": d.loc[fit, "target_month"].max(),
        }
    )
    return p, scales, diagnostics


def comparison_tables(ic):
    keys = ["target", "loss", "scope", "target_month"]
    b = ic[ic.input_set.eq(BASE)][keys + ["ic", "n"]].rename(
        columns={"ic": "baseline_ic", "n": "baseline_n"}
    )
    e = ic[ic.input_set.eq(EXT)][keys + ["ic", "n"]].rename(
        columns={"ic": "extended_ic", "n": "extended_n"}
    )
    paired = b.merge(e, on=keys, validate="one_to_one")
    assert paired.baseline_n.equals(paired.extended_n)
    paired["delta_ic"] = paired.extended_ic - paired.baseline_ic
    rows = []
    for (target, loss, scope), g in paired.groupby(["target", "loss", "scope"]):
        a, bm = mt.metrics(g.baseline_ic), mt.metrics(g.extended_ic)
        rows.append(
            dict(
                target=target,
                loss=loss,
                scope=scope,
                months=len(g),
                scored_rows=int(g.baseline_n.sum()),
                baseline_mean_ic=a["mean_ic"],
                extended_mean_ic=bm["mean_ic"],
                delta_mean_ic=float(g.delta_ic.mean()),
                baseline_icir=a["icir"],
                extended_icir=bm["icir"],
                delta_icir=bm["icir"] - a["icir"],
                delta_positive_month_share=float(g.delta_ic.gt(0).mean()),
            )
        )
    annual = (
        paired.assign(year=paired.target_month.str[:4])
        .groupby(["target", "loss", "scope", "year"])[
            ["baseline_ic", "extended_ic", "delta_ic"]
        ]
        .mean()
        .reset_index()
    )
    return pd.DataFrame(rows), paired, annual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--targets", nargs="+", choices=mt.TARGETS, default=list(mt.TARGETS)
    )
    args = parser.parse_args()
    root = Path.cwd()
    before = protected_hashes(root)
    started = datetime.now()
    t = time.perf_counter()
    # Audit all parquet reads and reject any final-test path before opening it.
    reads = []
    original = pd.read_parquet

    def guarded(path, *a, **kw):
        name = str(path).replace("\\", "/").lower()
        if any(s in name for s in ("test", "2022", "2023", "2024", "2025")):
            raise PermissionError("final_test_path_blocked:" + name)
        reads.append(str(path))
        return original(path, *a, **kw)

    pd.read_parquet = guarded
    try:
        prepared, config = prepare(root)
    finally:
        pd.read_parquet = original
    print(json.dumps(prepared["checks"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    parts = []
    scales = []
    diagnostics = []
    for fold in mt.plan_folds(config["MODEL_GRID"]):
        p, s, d = scale_fold(prepared, fold)
        scales.append(s)
        diagnostics.extend(d)
        parts.append(
            mt.run_linear(
                p,
                config,
                settings=[("ridge", x, "mse") for x in args.targets],
                sets=SETS,
                folds=[fold],
                workers=1,
                inner_scope="common",
                log=lambda s: print(s, flush=True),
            )
        )
    # These parts cover disjoint years; shared merge_results is for identical rows.
    result = {
        "val_keys": pd.concat([p["val_keys"] for p in parts], ignore_index=True),
        "predictions": {
            cid: np.concatenate([p["predictions"][cid] for p in parts])
            for cid in parts[0]["predictions"]
        },
        "selections": pd.concat([p["selections"] for p in parts], ignore_index=True),
        "inner_ic": pd.concat([p["inner_ic"] for p in parts], ignore_index=True),
        "timings": pd.concat([p["timings"] for p in parts], ignore_index=True),
        "stage1": pd.DataFrame(),
        "inner_scope": "common",
        "sets": SETS,
        "settings": [("ridge", x, "mse") for x in args.targets],
    }
    expected = mt.validation_rows(prepared, config)
    assert result["val_keys"].equals(expected)
    assert all(np.isfinite(v).all() for v in result["predictions"].values())
    ic = mt.outer_monthly_ic(result["val_keys"], result["predictions"])
    summary, paired, annual = comparison_tables(ic)
    assert paired.groupby(["target", "loss", "scope"]).size().eq(192).all()
    assert before == protected_hashes(root), "protected_files_changed"
    out = mt.save_run(
        root,
        config,
        prepared,
        result,
        folder_suffix="ridge",
        started=started,
        author="h4",
        experiment="h4_paired_oos",
        extra_tables={
            "paired_summary": (summary, "csv"),
            "paired_monthly_ic": (paired, "csv"),
            "paired_annual_ic": (annual, "csv"),
            "train_only_scales": (pd.concat(scales), "csv"),
            "design_diagnostics": (pd.DataFrame(diagnostics), "csv"),
        },
    )
    folder = Path(out["run_dir"])
    audit = dict(
        checks=prepared["checks"],
        parquet_reads=reads,
        final_test_access=False,
        primary_target="raw",
        sensitivity_targets=[x for x in args.targets if x != "raw"],
        folds=mt.plan_folds(config["MODEL_GRID"]),
        sets=SETS,
        preprocessing=(
            "H1/H3 monthly ranks; all design columns standardized on "
            "inner-fit rows only; missing predictions filled with scaled zero "
            "(train mean). No refit."
        ),
        interaction=(
            "Existing raw MOM times lagged EPU; raw MOM main effect not "
            "duplicated alongside existing MOM."
        ),
        icir="monthly mean IC / monthly sample standard deviation; not annualized",
        limitation=(
            "EPU historical release vintages unverified; one-month lag is an "
            "assumption. Development OOS, not final test."
        ),
        protected_hashes=before,
        seconds=time.perf_counter() - t,
        rerun="python -m src.h4.evaluation.ridge",
    )
    (folder / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mt.pack_result(folder)
    print(summary.to_string(index=False), flush=True)
    print("OUTPUT: " + str(folder), flush=True)


if __name__ == "__main__":
    main()
