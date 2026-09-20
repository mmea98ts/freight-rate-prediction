from __future__ import annotations

"""Reproducible exploratory analysis for the freight-rate assessment."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "posted_rate"
NUMERIC = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and save compact diagnostics.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent directory of src/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <project-dir>/analysis_outputs.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}")
    return path


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    data_dir = project_dir / "data"
    output_dir = (args.output_dir or project_dir / "analysis_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(require_file(data_dir / "train-test.csv"))
    validation = pd.read_csv(require_file(data_dir / "validation.csv"))
    template = pd.read_csv(require_file(data_dir / "validation-predictions-template.csv"))
    december = pd.read_csv(require_file(data_dir / "december-chart-inputs.csv"))

    for frame in (train, validation, december):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    datasets = {
        "train": train,
        "validation": validation,
        "template": template,
        "december": december,
    }

    quality_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for name, frame in datasets.items():
        quality_rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "duplicate_rows": int(frame.duplicated().sum()),
                "duplicate_load_ids": (
                    int(frame["load_id"].duplicated().sum())
                    if "load_id" in frame.columns
                    else np.nan
                ),
            }
        )
        for column, count in frame.isna().sum().items():
            if count:
                missing_rows.append(
                    {
                        "dataset": name,
                        "column": column,
                        "missing_count": int(count),
                        "missing_pct": float(100 * count / len(frame)),
                    }
                )

    quality = pd.DataFrame(quality_rows)
    missing = pd.DataFrame(missing_rows)
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    missing.to_csv(output_dir / "missing_values.csv", index=False)

    train["rate_per_mile"] = train[TARGET] / train["distance"]
    target_summary = train[TARGET].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    target_summary.rename("value").to_csv(output_dir / "target_summary.csv")

    correlations = train[NUMERIC + [TARGET]].corr(numeric_only=True)
    correlations.to_csv(output_dir / "numeric_correlations.csv")

    categorical_rows: list[dict[str, object]] = []
    for column in ["pickup", "delivery", "equipment"]:
        train_values = set(train[column].dropna().astype(str))
        validation_values = set(validation[column].dropna().astype(str))
        unseen = validation_values - train_values
        categorical_rows.append(
            {
                "feature": column,
                "train_unique": len(train_values),
                "validation_unique": len(validation_values),
                "unseen_categories": len(unseen),
                "validation_rows_with_unseen_category": int(
                    validation[column].astype(str).isin(unseen).sum()
                ),
                "unseen_values": ", ".join(sorted(unseen)),
            }
        )
    pd.DataFrame(categorical_rows).to_csv(
        output_dir / "categorical_coverage.csv", index=False
    )

    train_routes = set(zip(train["pickup"], train["delivery"]))
    validation_routes = list(zip(validation["pickup"], validation["delivery"]))
    unseen_route_rows = sum(route not in train_routes for route in validation_routes)

    monthly = (
        train.assign(month=train["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            loads=(TARGET, "size"),
            mean_rate=(TARGET, "mean"),
            median_rate=(TARGET, "median"),
            mean_rate_per_mile=("rate_per_mile", "mean"),
            mean_market_index=("market_index", "mean"),
        )
    )
    monthly.to_csv(output_dir / "monthly_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(train[TARGET], bins=70, color="#0B6477", alpha=0.9)
    ax.set(title="Posted-rate distribution", xlabel="Posted rate ($)", ylabel="Loads")
    save_figure(fig, output_dir / "target_distribution.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(train["distance"], train[TARGET], s=6, alpha=0.18, color="#0B6477")
    ax.set(title="Distance vs. posted rate", xlabel="Distance (miles)", ylabel="Posted rate ($)")
    save_figure(fig, output_dir / "distance_vs_posted_rate.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(train["rate_per_mile"], bins=80, color="#D97706", alpha=0.9)
    ax.set_xlim(0, min(6, float(train["rate_per_mile"].quantile(0.999))))
    ax.set(title="Rate-per-mile diagnostic", xlabel="Rate per mile ($)", ylabel="Loads")
    save_figure(fig, output_dir / "rate_per_mile_distribution.png")

    equipment = train.groupby("equipment")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    equipment.plot.barh(ax=ax, color="#0B6477")
    ax.set(title="Average posted rate by equipment", xlabel="Average posted rate ($)", ylabel="")
    save_figure(fig, output_dir / "average_rate_by_equipment.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(monthly["month"], monthly["mean_rate"], marker="o", color="#0B6477")
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Monthly mean posted rate", xlabel="Month", ylabel="Mean posted rate ($)")
    save_figure(fig, output_dir / "monthly_mean_posted_rate.png")

    report = f"""Freight Rate Assessment - EDA Summary
=====================================

Rows
- Development: {len(train):,}
- Final validation: {len(validation):,}
- Prediction template: {len(template):,}
- Fixed December scenario: {len(december):,}

Date coverage
- Development: {train['date'].min().date()} to {train['date'].max().date()}
- Final validation: {validation['date'].min().date()} to {validation['date'].max().date()}

Key findings
- Distance has a {train['distance'].corr(train[TARGET]):.3f} Pearson correlation with posted_rate.
- Development weight missing: {train['weight'].isna().sum():,}; negative: {(train['weight'] < 0).sum():,}.
- Development market_index missing: {train['market_index'].isna().sum():,}.
- Validation weight missing: {validation['weight'].isna().sum():,}; negative: {(validation['weight'] < 0).sum():,}.
- Validation market_index missing: {validation['market_index'].isna().sum():,}.
- Validation rows on routes absent from development: {unseen_route_rows:,} ({100 * unseen_route_rows / len(validation):.2f}%).
- Posted-rate median: ${train[TARGET].median():,.2f}; maximum: ${train[TARGET].max():,.2f}.
- Rate-per-mile median: ${train['rate_per_mile'].median():.3f}; maximum: ${train['rate_per_mile'].max():.3f}.

Modeling implications
- Use chronological validation because the final prediction window follows the development period.
- Correct negative weights, flag missing values, and let CatBoost handle remaining numeric missingness.
- Treat route endpoints and equipment as categorical features to accommodate unseen combinations.
- Use rate_per_mile as the model target, not as an input feature; convert predictions back to dollars with distance.
"""
    (output_dir / "eda_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved EDA outputs to: {output_dir}")


if __name__ == "__main__":
    main()
