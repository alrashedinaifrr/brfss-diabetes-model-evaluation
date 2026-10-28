"""Run the reproducible BRFSS cross-year diabetes classification analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW = ["DIABETE4", "_LLCPWT", "_AGEG5YR", "_BMI5", "SEXVAR",
       "EXERANY2", "SMOKE100", "EDUCA"]
FEATURES = ["age_group", "bmi", "sex", "exercise", "smoked100", "education"]
NO_AGE_FEATURES = [feature for feature in FEATURES if feature != "age_group"]
DEFAULT_THRESHOLDS = tuple(np.arange(0.05, 0.501, 0.05).round(2))
BMI_BINS = [-np.inf, 25, 30, np.inf]
BMI_LABELS = ["BMI <25", "BMI 25-29.9", "BMI >=30"]


def transform(chunk):
    """Select valid self-reported diabetes records and clean predictors."""
    required = set(RAW)
    missing = required - set(chunk.columns)
    if missing:
        raise ValueError(f"BRFSS chunk is missing required fields: {sorted(missing)}")
    d = chunk.loc[chunk.DIABETE4.isin([1, 3]) & chunk._LLCPWT.gt(0), RAW].copy()
    d["target"] = (d.DIABETE4 == 1).astype(int)
    d["age_group"] = d._AGEG5YR.where(d._AGEG5YR.between(1, 13))
    d["bmi"] = (d._BMI5 / 100).where((d._BMI5 / 100).between(10, 80))
    for raw, new, valid in [
        ("SEXVAR", "sex", [1, 2]),
        ("EXERANY2", "exercise", [1, 2]),
        ("SMOKE100", "smoked100", [1, 2]),
        ("EDUCA", "education", [1, 2, 3, 4, 5, 6]),
    ]:
        d[new] = d[raw].where(d[raw].isin(valid))
    return d[["target", *FEATURES, "_LLCPWT"]]


def load_xpt(path, chunksize=40000):
    """Read a CDC XPT file in chunks to bound peak parsing memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing BRFSS XPT file: {path}")
    chunks = pd.read_sas(path, format="xport", encoding="utf-8", chunksize=chunksize)
    return pd.concat((transform(chunk) for chunk in chunks), ignore_index=True)


def model_for(strategy="median", features=FEATURES):
    """Build a logistic-regression pipeline for a selected feature set."""
    features = list(features)
    numeric = [feature for feature in features if feature == "bmi"]
    categorical = [feature for feature in features if feature != "bmi"]
    if not features or not numeric:
        raise ValueError("the model requires the numeric BMI predictor")
    if strategy == "complete_case":
        numeric_pipe = Pipeline([("scale", StandardScaler())])
        categorical_pipe = OneHotEncoder(handle_unknown="ignore")
    else:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=(strategy == "indicator"))),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
    transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("pre", transformer),
        ("lr", LogisticRegression(max_iter=500, solver="lbfgs")),
    ])


def weighted_decision_analysis(y, probability, weights, thresholds):
    """Return weighted classification metrics and decision-curve net benefit.

    The thresholds are exploratory scenarios unless an intervention, its
    benefits and harms, and service capacity justify a particular cutoff.
    Survey weights yield weighted point estimates here, not design-correct
    standard errors or confidence intervals.
    """
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or weights.shape != y.shape:
        raise ValueError("y, probability, and weights must be one-dimensional arrays of equal length")
    if len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain at least one binary observation")
    y = y.astype(int)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("probability must contain finite values between 0 and 1")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("weights must be finite and positive")
    if (thresholds.ndim != 1 or len(thresholds) == 0 or not np.isfinite(thresholds).all()
            or (thresholds <= 0).any() or (thresholds >= 1).any()):
        raise ValueError("thresholds must be a non-empty vector with values strictly between 0 and 1")

    total_weight = weights.sum()
    prevalence = np.average(y, weights=weights)
    rows = []
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = weights[(y == 1) & predicted].sum()
        fp = weights[(y == 0) & predicted].sum()
        tn = weights[(y == 0) & ~predicted].sum()
        fn = weights[(y == 1) & ~predicted].sum()
        odds = threshold / (1 - threshold)
        rows.append({
            "threshold": float(threshold),
            "weighted_sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "weighted_specificity": float(tn / (tn + fp)) if tn + fp else None,
            "weighted_precision": float(tp / (tp + fp)) if tp + fp else None,
            "weighted_npv": float(tn / (tn + fn)) if tn + fn else None,
            "weighted_flagged_fraction": float((tp + fp) / total_weight),
            "net_benefit_model": float((tp - fp * odds) / total_weight),
            "net_benefit_treat_all": float(prevalence - (1 - prevalence) * odds),
            "net_benefit_treat_none": 0.0,
        })
    return rows


def weighted_age_auc_decomposition(y, score, weight, age_group):
    """Exactly partition weighted AUROC into same-age and different-age pairs."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    age_group = np.asarray(age_group)
    if not (y.ndim == score.ndim == weight.ndim == age_group.ndim == 1):
        raise ValueError("all decomposition inputs must be one-dimensional")
    if not (len(y) == len(score) == len(weight) == len(age_group)):
        raise ValueError("decomposition inputs must have equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only binary observations")
    y = y.astype(int)
    valid = (np.isin(y, [0, 1]) & np.isfinite(score) & np.isfinite(weight)
             & (weight > 0) & pd.notna(age_group))
    y, score, weight, age_group = y[valid], score[valid], weight[valid], age_group[valid]
    if len(y) == 0 or np.unique(y).size != 2:
        return {"overall_auc": None, "within_age_pair_share": None,
                "within_age_auc": None, "between_age_pair_share": None,
                "between_age_auc": None, "reconstruction_error": None}

    overall_auc = float(roc_auc_score(y, score, sample_weight=weight))
    positive_weight = weight[y == 1].sum()
    negative_weight = weight[y == 0].sum()
    total_pairs = positive_weight * negative_weight
    within_pairs = 0.0
    within_wins = 0.0
    for age in pd.unique(age_group):
        selected = age_group == age
        ya, sa, wa = y[selected], score[selected], weight[selected]
        pos = wa[ya == 1].sum()
        neg = wa[ya == 0].sum()
        if pos <= 0 or neg <= 0:
            continue
        pair_mass = pos * neg
        within_pairs += pair_mass
        within_wins += pair_mass * roc_auc_score(ya, sa, sample_weight=wa)

    between_pairs = max(total_pairs - within_pairs, 0.0)
    within_auc = within_wins / within_pairs if within_pairs > 0 else None
    between_auc = (
        (overall_auc * total_pairs - within_wins) / between_pairs
        if between_pairs > 0 else None
    )
    within_share = within_pairs / total_pairs
    between_share = between_pairs / total_pairs
    reconstructed = None
    error = None
    if within_auc is not None and between_auc is not None:
        reconstructed = within_share * within_auc + between_share * between_auc
        error = abs(reconstructed - overall_auc)
        if error > 1e-9:
            raise ArithmeticError("age-pair AUROC decomposition failed reconstruction")
    return {
        "overall_auc": overall_auc,
        "within_age_pair_share": float(within_share),
        "within_age_auc": float(within_auc) if within_auc is not None else None,
        "between_age_pair_share": float(between_share),
        "between_age_auc": float(between_auc) if between_auc is not None else None,
        "reconstruction_error": float(error) if error is not None else None,
    }


def _metric_row(model_name, strategy, fitdata, cohort, data, probability):
    y = data.target.to_numpy(dtype=int)
    weights = data._LLCPWT.to_numpy(dtype=float)
    return {
        "model": model_name,
        "strategy": strategy,
        "cohort": cohort,
        "train_n": int(len(fitdata)),
        "test_n": int(len(data)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "weighted_auroc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "weighted_auprc": float(average_precision_score(y, probability, sample_weight=weights)),
        "weighted_brier": float(brier_score_loss(y, probability, sample_weight=weights)),
        "weighted_log_loss": float(log_loss(y, probability, sample_weight=weights)),
        "mean_pred": float(probability.mean()),
        "observed_prevalence": float(y.mean()),
        "weighted_mean_pred": float(np.average(probability, weights=weights)),
        "weighted_observed_prevalence": float(np.average(y, weights=weights)),
    }


def _subgroup_outputs(test, predictions):
    analysis = test.loc[
        test.age_group.between(1, 13) & test.bmi.notna() & test._LLCPWT.gt(0)
    ].copy()
    analysis["bmi_group"] = pd.cut(
        analysis.bmi, bins=BMI_BINS, labels=BMI_LABELS, right=False
    )
    subgroup_rows = []
    decompositions = {}
    for model_name, probability in predictions.items():
        analysis["probability"] = probability[analysis.index]
        for bmi_group, group in analysis.groupby("bmi_group", observed=True):
            y = group.target.to_numpy(dtype=int)
            weights = group._LLCPWT.to_numpy(dtype=float)
            auc = (float(roc_auc_score(y, group.probability, sample_weight=weights))
                   if np.unique(y).size == 2 else None)
            subgroup_rows.append({
                "model": model_name,
                "bmi_group": str(bmi_group),
                "n": int(len(group)),
                "cases": int(y.sum()),
                "weighted_prevalence": float(np.average(y, weights=weights)),
                "weighted_auroc": auc,
            })
            if bmi_group in ("BMI <25", "BMI >=30"):
                decompositions[(model_name, str(bmi_group))] = weighted_age_auc_decomposition(
                    y, group.probability.to_numpy(), weights, group.age_group.to_numpy()
                )

    gap_rows = []
    for model_name in predictions:
        low = decompositions.get((model_name, "BMI <25"))
        high = decompositions.get((model_name, "BMI >=30"))
        if not low or not high or any(
            low.get(key) is None or high.get(key) is None
            for key in ("within_age_auc", "between_age_auc")
        ):
            continue
        within_disc = (low["within_age_pair_share"] + high["within_age_pair_share"]) / 2 * (
            low["within_age_auc"] - high["within_age_auc"]
        )
        between_disc = (low["between_age_pair_share"] + high["between_age_pair_share"]) / 2 * (
            low["between_age_auc"] - high["between_age_auc"]
        )
        within_comp = (low["within_age_pair_share"] - high["within_age_pair_share"]) * (
            low["within_age_auc"] + high["within_age_auc"]
        ) / 2
        between_comp = (low["between_age_pair_share"] - high["between_age_pair_share"]) * (
            low["between_age_auc"] + high["between_age_auc"]
        ) / 2
        observed_gap = low["overall_auc"] - high["overall_auc"]
        reconstructed_gap = within_disc + between_disc + within_comp + between_comp
        gap_rows.append({
            "model": model_name,
            "comparison": "BMI <25 minus BMI >=30",
            "observed_gap": observed_gap,
            "within_age_discrimination": within_disc,
            "between_age_discrimination": between_disc,
            "within_age_composition": within_comp,
            "between_age_composition": between_comp,
            "reconstructed_gap": reconstructed_gap,
            "reconstruction_error": abs(observed_gap - reconstructed_gap),
        })
        if abs(observed_gap - reconstructed_gap) > 1e-9:
            raise ArithmeticError("BMI-group AUROC gap failed exact decomposition")
    return pd.DataFrame(subgroup_rows), pd.DataFrame(gap_rows)


def _calibration_deciles(test, predictions):
    rows = []
    for model_name, probability in predictions.items():
        order = np.argsort(probability, kind="mergesort")
        bins = np.empty(len(order), dtype=int)
        bins[order] = np.minimum(np.arange(len(order)) * 10 // max(len(order), 1), 9)
        for decile in range(10):
            mask = bins == decile
            if not mask.any():
                continue
            weights = test._LLCPWT.to_numpy(dtype=float)[mask]
            y = test.target.to_numpy(dtype=float)[mask]
            p = probability[mask]
            rows.append({
                "model": model_name,
                "decile": decile + 1,
                "n": int(mask.sum()),
                "weighted_predicted_risk": float(np.average(p, weights=weights)),
                "weighted_observed_prevalence": float(np.average(y, weights=weights)),
                "weight_sum": float(weights.sum()),
            })
    return pd.DataFrame(rows)


def evaluate(train, test, decision_thresholds=DEFAULT_THRESHOLDS):
    """Fit the full model, age ablation, training-weight sensitivity, and checks."""
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    common = test.loc[test[FEATURES].notna().all(axis=1)].copy()
    rows = []
    predictions = {}
    specifications = [
        ("full", "median", FEATURES, train, False),
        ("full", "indicator", FEATURES, train, False),
        ("full", "complete_case", FEATURES, train.dropna(subset=FEATURES), False),
        ("without_age", "median", NO_AGE_FEATURES, train, False),
        ("weighted_training", "median", FEATURES, train, True),
    ]
    for model_name, strategy, features, fitdata, use_training_weights in specifications:
        model = model_for(strategy, features)
        fit_kwargs = {}
        if use_training_weights:
            # Normalize to mean one so regularization does not depend on the
            # arbitrary scale of the survey weights.
            training_weights = fitdata._LLCPWT.to_numpy(dtype=float)
            fit_kwargs["lr__sample_weight"] = training_weights / training_weights.mean()
        model.fit(fitdata[features], fitdata.target, **fit_kwargs)
        if strategy == "complete_case":
            cohorts = [("common_complete_case", common)]
        else:
            eligible_probability = model.predict_proba(test[features])[:, 1]
            if model_name in ("full", "without_age", "weighted_training"):
                if model_name not in predictions or strategy == "median":
                    predictions[model_name] = eligible_probability
            cohorts = [("all_eligible", test), ("common_complete_case", common)]
        for cohort, data in cohorts:
            probability = (
                eligible_probability if cohort == "all_eligible"
                else model.predict_proba(data[features])[:, 1]
            )
            rows.append(_metric_row(model_name, strategy, fitdata, cohort, data, probability))

    decision_rows = []
    for model_name, probability in predictions.items():
        for item in weighted_decision_analysis(
            test.target.to_numpy(), probability, test._LLCPWT.to_numpy(), decision_thresholds
        ):
            decision_rows.append({"model": model_name, **item})
    subgroup_table, gap_table = _subgroup_outputs(test, predictions)
    calibration_table = _calibration_deciles(test, predictions)
    return {
        "protocol": "BRFSS 2024 training; BRFSS 2025 evaluation, reused during development; cross-sectional self-reported prevalence classification",
        "n_train_2024": int(len(train)),
        "n_test_2025": int(len(test)),
        "n_common_test_2025": int(len(common)),
        "results": rows,
        "decision_analysis": decision_rows,
        "bmi_subgroup_auroc": subgroup_table.to_dict(orient="records"),
        "bmi_gap_decomposition": gap_table.to_dict(orient="records"),
        "calibration_deciles": calibration_table.to_dict(orient="records"),
        "decision_analysis_note": "Thresholds are scenario comparisons, not recommended clinical cutoffs. Net benefit does not account for design variance, clinical harms, or service capacity.",
        "inference_note": "Survey-weighted point estimates only; no design-correct standard errors or confidence intervals. BMI gap decomposition is an exact descriptive identity, not a causal or fairness result.",
        "limitations": [
            "2025 was reused during model development and is not an untouched final holdout",
            "the outcome is self-reported prevalent diabetes, not future incidence or clinical diagnosis",
            "survey-design-correct uncertainty has not been established",
            "age ablation is descriptive and not proof of fairness improvement",
            "no prospective clinical validation or decision-utility evidence",
        ],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def save_figures(result, output_dir):
    """Create portfolio figures from the exact tables written by the run."""
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(result["results"])
    models = ["full", "without_age", "weighted_training"]
    eligible = metrics.loc[
        (metrics.cohort == "all_eligible") & metrics.model.isin(models)
    ].drop_duplicates("model").set_index("model").reindex(models).dropna(how="all")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Full", "Without age", "Weighted fit"][:len(eligible)]
    x = np.arange(len(eligible))
    axes[0].bar(x, eligible.weighted_auroc, color=["#2b6cb0", "#ed8936", "#38a169"][:len(eligible)])
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Survey-weighted AUROC")
    axes[1].bar(x, eligible.weighted_brier, color=["#2b6cb0", "#ed8936", "#38a169"][:len(eligible)])
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_title("Survey-weighted Brier score")
    fig.suptitle("BRFSS 2025 evaluation — descriptive model comparison")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    subgroups = pd.DataFrame(result["bmi_subgroup_auroc"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name in models:
        group = subgroups[subgroups.model == model_name]
        if not group.empty:
            ax.plot(group.bmi_group, group.weighted_auroc, marker="o", label=model_name)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Survey-weighted AUROC")
    ax.set_title("Discrimination by BMI subgroup")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "bmi_subgroup_auroc_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    calibration = pd.DataFrame(result["calibration_deciles"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model_name in models:
        group = calibration[calibration.model == model_name]
        if not group.empty:
            ax.plot(group.weighted_predicted_risk, group.weighted_observed_prevalence,
                    marker="o", label=model_name)
    lim = max(0.1, float(calibration[["weighted_predicted_risk", "weighted_observed_prevalence"]].max().max()) * 1.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Weighted mean predicted risk",
           ylabel="Weighted observed prevalence", title="Weighted calibration by risk decile")
    ax.legend(title="Model")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "weighted_calibration_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    decisions = pd.DataFrame(result["decision_analysis"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in models:
        group = decisions[decisions.model == model_name]
        if not group.empty:
            ax.plot(group.threshold, group.net_benefit_model, marker="o", label=model_name)
    baseline = decisions[decisions.model == "full"]
    if not baseline.empty:
        ax.plot(baseline.threshold, baseline.net_benefit_treat_all,
                linestyle="--", color="gray", label="Treat all")
    ax.axhline(0, linestyle=":", color="black", label="Treat none")
    ax.set(xlabel="Exploratory threshold", ylabel="Weighted net benefit",
           title="Decision-curve scenarios (not clinical recommendations)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "decision_curve_2025.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing LLCP2024.XPT and LLCP2025.XPT")
    parser.add_argument("--output-dir", type=Path, default=Path("results_v11"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                        help="Exploratory decision-curve thresholds; justify before operational use")
    args = parser.parse_args()
    train_path = args.data_dir / "LLCP2024.XPT"
    test_path = args.data_dir / "LLCP2025.XPT"
    train, test = load_xpt(train_path), load_xpt(test_path)
    result = evaluate(train, test, args.thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_2025.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output_tables = {
        "model_comparison_2025.csv": result["results"],
        "decision_analysis_2025.csv": result["decision_analysis"],
        "bmi_subgroup_auroc_2025.csv": result["bmi_subgroup_auroc"],
        "bmi_gap_decomposition_2025.csv": result["bmi_gap_decomposition"],
        "calibration_deciles_2025.csv": result["calibration_deciles"],
    }
    for filename, rows in output_tables.items():
        pd.DataFrame(rows).to_csv(args.output_dir / filename, index=False)
    save_figures(result, args.output_dir)
    manifest = {
        "training_file": {"name": train_path.name, "sha256": _sha256(train_path), "rows": int(len(train))},
        "evaluation_file": {"name": test_path.name, "sha256": _sha256(test_path), "rows": int(len(test))},
        "python": platform.python_version(),
        "package_versions": {name: _package_version(name) for name in
                              ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
        "thresholds": list(map(float, args.thresholds)),
        "method_note": "Point estimates and descriptive bootstrap-free decomposition only; no design-correct uncertainty.",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Analysis complete: {args.output_dir}")
    print(f"Training N={len(train):,}; evaluation N={len(test):,}; subgroup tables and four figures saved.")


if __name__ == "__main__":
    main()
