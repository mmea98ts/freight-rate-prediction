from __future__ import annotations

"""Validate the final engineered-RPM model with chronological holdouts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_predict import (
    CATEGORICAL_FEATURES,
    TARGET,
    build_model,
    clean_and_engineer,
    derive_final_rpm_bounds,
    prepare_model_x,
)


FOLDS = {
    "August": (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")),
    "September": (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")),
    "October": (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chronological validation for the final engineered-RPM model."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent directory of src/.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 100 iterations for a fast execution check.",
    )
    return parser.parse_args()


def metrics(y_true: pd.Series, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, prediction)),
        float(np.sqrt(mean_squared_error(y_true, prediction))),
        float(r2_score(y_true, prediction)),
    )


def main() -> None:
    args = parse_args()
    source = args.project_dir.resolve() / "data" / "train-test.csv"
    if not source.is_file():
        raise SystemExit(f"Required file not found: {source}")

    data = clean_and_engineer(pd.read_csv(source))
    rows: list[dict[str, float | int | str]] = []

    print("Chronological validation of the final engineered-RPM model")
    print("-" * 78)

    for fold, (start, end) in FOLDS.items():
        train_raw = data.loc[data["date"] < start].copy()
        valid = data.loc[data["date"].between(start, end, inclusive="left")].copy()

        lower, upper = derive_final_rpm_bounds(train_raw)
        train_rpm = train_raw[TARGET] / train_raw["distance"]
        train = train_raw.loc[train_rpm.between(lower, upper)].copy()

        model = build_model(iterations=100 if args.quick else None)
        model.fit(
            prepare_model_x(train),
            train[TARGET] / train["distance"],
            cat_features=CATEGORICAL_FEATURES,
        )

        predicted_rpm = model.predict(prepare_model_x(valid))
        prediction = np.asarray(predicted_rpm, dtype=float) * valid[
            "distance"
        ].to_numpy(dtype=float)
        prediction = np.maximum(prediction, 1.0)

        mae, rmse, r2 = metrics(valid[TARGET], prediction)
        valid_rpm = valid[TARGET] / valid["distance"]
        normal = valid_rpm.between(lower, upper).to_numpy()
        normal_mae, _, _ = metrics(valid.loc[normal, TARGET], prediction[normal])

        rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "excluded_labels": len(train_raw) - len(train),
                "validation_rows": len(valid),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "normal_mae": normal_mae,
            }
        )
        print(
            f"{fold:9s} | MAE=${mae:8.2f} | RMSE=${rmse:8.2f} | "
            f"R2={r2:.4f} | normal-label MAE=${normal_mae:7.2f}"
        )

    results = pd.DataFrame(rows)
    print("-" * 78)
    print(
        "Mean      | "
        f"MAE=${results['mae'].mean():8.2f} | "
        f"RMSE=${results['rmse'].mean():8.2f} | "
        f"R2={results['r2'].mean():.4f} | "
        f"normal-label MAE=${results['normal_mae'].mean():7.2f}"
    )
    print("Validation uses only earlier months to predict each held-out month.")


if __name__ == "__main__":
    main()
