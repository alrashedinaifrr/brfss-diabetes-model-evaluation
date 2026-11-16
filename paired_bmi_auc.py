"""Estimate paired, stratified BRFSS AUROC contrasts for two fitted models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import __version__ as sklearn_version
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from run_v11 import FEATURES, NO_AGE_FEATURES, RAW, model_for, transform


DESIGN = ["_STSTR", "_PSU"]
DOMAIN_MODELS = {"full": FEATURES, "without_age": NO_AGE_FEATURES}


def read_fields(path, columns):
    """Keep the public-use design identifiers on the same rows as scores."""
    chunks = pd.read_sas(path, format="xport", chunksize=12_000)
    with chunks:
        return pd.concat((chunk[columns] for chunk in chunks), ignore_index=True)


def auc_linearization(y, probability, weights):
    """Return weighted concordance and each row's weighted derivative."""
    _, rank = np.unique(probability, return_inverse=True)
    cases = y == 1
    pos = np.bincount(rank, weights=weights * cases)
    neg = np.bincount(rank, weights=weights * ~cases)
    pos_total, neg_total = pos.sum(), neg.sum()
    if min(pos_total, neg_total) <= 0:
        raise ValueError("Both outcome classes need positive domain weight")
    pos_placement = (neg.cumsum() - 0.5 * neg)[rank] / neg_total
    neg_placement = (pos[::-1].cumsum()[::-1] - 0.5 * pos)[rank] / pos_total
    auc = np.average(pos_placement[cases], weights=weights[cases])
    placement = np.where(cases, pos_placement, neg_placement)
    denominators = np.where(cases, pos_total, neg_total)
    return auc, weights * (placement - auc) / denominators


def stratified_covariance(identifiers, contributions):
    """Estimate covariance from centered, nested PSU totals under a WR approximation."""
    check = identifiers.to_numpy()
    if not np.isfinite(check).all() or (check <= 0).any():
        raise ValueError("Missing or invalid public-use design identifiers")
    psu = identifiers.join(contributions).groupby(DESIGN, sort=False).sum()
    within = psu.groupby(level="_STSTR", sort=False)
    sizes = within.size()
    centered = psu - within.transform("mean")
    counts = sizes.reindex(psu.index.get_level_values("_STSTR")).to_numpy()
    ordinary = counts > 1
    scaled = centered.to_numpy()[ordinary] * np.sqrt(counts[ordinary] / (counts[ordinary] - 1))[:, None]
    multi_psu_cov = scaled.T @ scaled
    # Center singleton PSU totals on the all-PSU mean (survey's "adjust" rule).
    singleton_residual = psu.to_numpy()[~ordinary] - psu.mean().to_numpy()
    adjusted_cov = multi_psu_cov + singleton_residual.T @ singleton_residual
    # Sensitivity check: assign the average non-singleton stratum contribution.
    average_cov = multi_psu_cov * len(sizes) / sizes.gt(1).sum()
    return adjusted_cov, average_cov, sizes


def check_score_derivative():
    """Check ties and the paired-weight derivative against independent AUC calls."""
    target = np.array([1, 0, 1, 0, 1, 0, 0])
    scores = np.array([0.1, 0.1, 0.5, 0.3, 0.3, 0.9, 0.2])
    weights = np.array([5.0, 1.0, 2.5, 7.0, 3.5, 2.0, 2.0])
    direction = np.array([0.2, 0.7, -0.1, -0.1, 0.9, -0.2, 0.3])
    auc, derivative = auc_linearization(target, scores, weights)
    np.testing.assert_allclose(auc, roc_auc_score(target, scores, sample_weight=weights), atol=1e-14)
    eps = 1e-5
    plus = roc_auc_score(target, scores, sample_weight=weights * np.exp(eps * direction))
    minus = roc_auc_score(target, scores, sample_weight=weights * np.exp(-eps * direction))
    np.testing.assert_allclose(derivative @ direction, (plus - minus) / (2 * eps), atol=1e-10)
    np.testing.assert_allclose(derivative.sum(), 0, atol=1e-14)


def main(training_xpt, evaluation_xpt, output_path):
    check_score_derivative()
    train_raw = read_fields(training_xpt, RAW)
    train = transform(train_raw)
    print(f"2024: {len(train_raw)} source rows, {len(train)} eligible", flush=True)
    test_raw = read_fields(evaluation_xpt, RAW + DESIGN)
    test = transform(test_raw)
    print(f"2025: {len(test_raw)} source rows, {len(test)} eligible", flush=True)
    if (len(train_raw), len(train), len(test_raw), len(test)) != (457670, 441934, 356158, 342539):
        raise ValueError("Cohort counts differ from the 2024 and 2025 releases used in the portfolio")

    scores = {}
    metrics = {}
    for model, features in DOMAIN_MODELS.items():
        fit = model_for(features=features).fit(train[features], train.target)
        score = fit.predict_proba(test[features])[:, 1]
        scores[model] = pd.Series(score, index=test.index)
        metrics[model] = {
            "auc": roc_auc_score(test.target, score, sample_weight=test._LLCPWT),
            "brier": brier_score_loss(test.target, score, sample_weight=test._LLCPWT),
        }
    complete = test.loc[test.age_group.notna() & test.bmi.notna()]
    groups = {"low": complete.loc[complete.bmi.lt(25)], "high": complete.loc[complete.bmi.ge(30)]}
    counts = {label: len(group) for label, group in groups.items()}
    if len(complete) != 307809 or counts != {"low": 97914, "high": 100215}:
        raise ValueError("Evaluation domain sizes differ from the portfolio's BMI cohorts")

    design = test_raw.loc[test_raw._LLCPWT.gt(0), DESIGN]
    point = {}
    contributions = pd.DataFrame(index=design.index)
    for label, group in groups.items():
        for model, probability in scores.items():
            name = f"{label}_{model}"
            y = group.target.to_numpy()
            w = group._LLCPWT.to_numpy()
            p = probability.loc[group.index].to_numpy()
            point[name], derivative = auc_linearization(y, p, w)
            np.testing.assert_allclose(point[name], roc_auc_score(y, p, sample_weight=w), atol=1e-13)
            # Domains use the parent design, with zero terms outside each BMI group.
            contributions[name] = pd.Series(derivative, index=group.index).reindex(design.index, fill_value=0)
    print(f"Point estimates: {point}", flush=True)
    np.testing.assert_allclose(contributions.sum(), 0, atol=1e-12)
    covariance, average_covariance, strata_sizes = stratified_covariance(design, contributions)

    contrasts = {
        "full_gap": {"low_full": 1, "high_full": -1},
        "without_age_gap": {"low_without_age": 1, "high_without_age": -1},
        "gap_reduction": {"low_full": 1, "high_full": -1, "low_without_age": -1, "high_without_age": 1},
        "low_change": {"low_full": -1, "low_without_age": 1},
        "high_change": {"high_full": -1, "high_without_age": 1},
    }
    intervals = {}
    for name, contrast in contrasts.items():
        coefficients = np.array([contrast.get(column, 0) for column in contributions])
        estimate = coefficients @ np.array([point[column] for column in contributions])
        se = np.sqrt(coefficients @ covariance @ coefficients)
        intervals[name] = {
            "estimate": estimate,
            "se": se,
            "ci95": (estimate + norm.ppf([0.025, 0.975]) * se).tolist(),
            "ci95_singleton_average": (estimate + norm.ppf([0.025, 0.975]) * np.sqrt(coefficients @ average_covariance @ coefficients)).tolist(),
        }

    output = {
        "sklearn": sklearn_version,
        "overall": metrics,
        "domain_counts": counts,
        "auroc": point,
        "contrasts": intervals,
        "design_records": len(design),
        "strata": len(strata_sizes),
        "nested_psus": int(strata_sizes.sum()),
        "singleton_strata": int(strata_sizes.eq(1).sum()),
        "min_psus": int(strata_sizes.min()),
        "max_psus": int(strata_sizes.max()),
        "relative_reduction": intervals["gap_reduction"]["estimate"] / intervals["full_gap"]["estimate"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-xpt", required=True, type=Path, help="Uncompressed 2024 CDC XPT file")
    parser.add_argument("--evaluation-xpt", required=True, type=Path, help="Uncompressed 2025 CDC XPT file")
    parser.add_argument("--output", type=Path, default=Path("brfss_paired_design_results.json"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.training_xpt, args.evaluation_xpt, args.output)
