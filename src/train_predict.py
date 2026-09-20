from __future__ import annotations

"""Train the final engineered-RPM model and create submission predictions.

The model predicts rate per mile and converts it back to a dollar rate by
multiplying by distance. Only information available at prediction time is used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


TARGET = "posted_rate"
DATE_COL = "date"
DATE_ORIGIN = pd.Timestamp("2025-01-01")

MAD_SCALE = 1.4826
MAD_MULTIPLIER = 5.0

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "month",
    "day_of_week",
    "route",
    "route_equipment",
    "pickup_equipment",
    "delivery_equipment",
    "distance_band",
]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "weight_was_negative",
    "weight_missing",
    "day_of_month",
    "day_of_year",
    "days_since_start",
    "market_index",
    "market_index_missing",
    "log_distance",
    "sqrt_distance",
    "inverse_distance",
    "distance_squared",
    "weight_per_mile",
    "market_distance_interaction",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FINAL_MODEL_CONFIG = {
    "iterations": 700,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 8.0,
    "loss_function": "RMSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the engineered-RPM model and create final predictions."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent directory of src/.",
    )
    return parser.parse_args()


def read_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Could not read {label}: {exc}") from exc


def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply identical, deployment-safe preprocessing to any input frame."""
    df = frame.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("Invalid date value found.")

    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight_missing"] = df["weight"].isna().astype(int)
    df.loc[df["weight"] < 0, "weight"] = df.loc[
        df["weight"] < 0, "weight"
    ].abs()

    df["market_index_missing"] = df["market_index"].isna().astype(int)

    df["month"] = df[DATE_COL].dt.month.astype(str)
    df["day_of_week"] = df[DATE_COL].dt.dayofweek.astype(str)
    df["day_of_month"] = df[DATE_COL].dt.day.astype(int)
    df["day_of_year"] = df[DATE_COL].dt.dayofyear.astype(int)
    df["days_since_start"] = (df[DATE_COL] - DATE_ORIGIN).dt.days.astype(int)

    pickup = df["pickup"].fillna("__MISSING__").astype(str)
    delivery = df["delivery"].fillna("__MISSING__").astype(str)
    equipment = df["equipment"].fillna("__MISSING__").astype(str)

    df["route"] = pickup + "__TO__" + delivery
    df["route_equipment"] = df["route"] + "__" + equipment
    df["pickup_equipment"] = pickup + "__" + equipment
    df["delivery_equipment"] = delivery + "__" + equipment

    df["distance_band"] = pd.cut(
        df["distance"],
        bins=[0, 250, 500, 750, 1000, 1500, 2000, 3000, np.inf],
        labels=[
            "0-250",
            "251-500",
            "501-750",
            "751-1000",
            "1001-1500",
            "1501-2000",
            "2001-3000",
            "3000+",
        ],
        include_lowest=True,
    ).astype(str)

    safe_distance = df["distance"].clip(lower=1e-6)
    df["log_distance"] = np.log1p(safe_distance)
    df["sqrt_distance"] = np.sqrt(safe_distance)
    df["inverse_distance"] = 1.0 / safe_distance
    df["distance_squared"] = safe_distance**2
    df["weight_per_mile"] = df["weight"] / safe_distance
    df["market_distance_interaction"] = df["market_index"] * df["distance"]

    month_num = df[DATE_COL].dt.month.astype(float)
    dow_num = df[DATE_COL].dt.dayofweek.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow_num / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow_num / 7.0)
    df["is_weekend"] = (dow_num >= 5).astype(int)

    return df


def derive_final_rpm_bounds(train_df: pd.DataFrame) -> tuple[float, float]:
    """Learn robust label-quality bounds from labeled development data only."""
    rpm = train_df[TARGET] / train_df["distance"]
    if (rpm <= 0).any():
        raise ValueError("Non-positive rate-per-mile found.")

    log_rpm = np.log(rpm.to_numpy(dtype=float))
    median_log = float(np.median(log_rpm))
    mad_log = float(np.median(np.abs(log_rpm - median_log)))
    if mad_log == 0:
        raise ValueError("MAD is zero; cannot derive robust RPM bounds.")

    robust_sigma = MAD_SCALE * mad_log
    return (
        float(np.exp(median_log - MAD_MULTIPLIER * robust_sigma)),
        float(np.exp(median_log + MAD_MULTIPLIER * robust_sigma)),
    )


def prepare_model_x(frame: pd.DataFrame) -> pd.DataFrame:
    model_x = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        model_x[column] = model_x[column].fillna("__MISSING__").astype(str)
    return model_x


def build_model(iterations: int | None = None) -> CatBoostRegressor:
    config = dict(FINAL_MODEL_CONFIG)
    if iterations is not None:
        config["iterations"] = iterations
    return CatBoostRegressor(
        **config,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def enforce_positive(predictions: np.ndarray, label: str) -> np.ndarray:
    pred = np.asarray(predictions, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError(f"{label} contains non-finite predictions.")
    if (pred <= 0).any():
        print(
            f"WARNING: {int((pred <= 0).sum()):,} {label} predictions "
            "were non-positive and were clipped to $1.00."
        )
        pred = np.maximum(pred, 1.0)
    return pred


def build_daily_december_market_index(validation_raw: pd.DataFrame) -> pd.Series:
    """Derive each December day's median market context from validation data."""
    temp = validation_raw.copy()
    temp[DATE_COL] = pd.to_datetime(temp[DATE_COL], errors="coerce")
    december = temp.loc[
        temp[DATE_COL].between("2025-12-01", "2025-12-31", inclusive="both")
    ].copy()
    if december.empty:
        raise ValueError("validation.csv contains no December rows.")

    expected_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    daily = december.groupby(DATE_COL)["market_index"].median().reindex(expected_dates)
    daily = daily.interpolate(method="linear", limit_direction="both")
    if daily.isna().any():
        fallback = float(pd.to_numeric(december["market_index"], errors="coerce").median())
        daily = daily.fillna(fallback)
    if daily.isna().any():
        raise ValueError("Unable to derive market_index for all December dates.")
    return daily


def validate_input_schemas(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    template: pd.DataFrame,
    december: pd.DataFrame,
) -> None:
    train_required = {
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        TARGET,
    }
    validation_required = {
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
    }
    december_required = {
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    }

    for label, frame, required in [
        ("train-test.csv", train, train_required),
        ("validation.csv", validation, validation_required),
        ("december-chart-inputs.csv", december, december_required),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"{label} missing columns: {', '.join(sorted(missing))}")

    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise SystemExit(
            "validation-predictions-template.csv must contain exactly "
            "load_id,predicted_rate in that order."
        )


def format_rate(value: float) -> str:
    """Keep the submission column numeric-looking with exactly four decimals."""
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    input_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_required_csv(input_dir / "train-test.csv", "train-test.csv")
    validation_raw = read_required_csv(input_dir / "validation.csv", "validation.csv")
    template = read_required_csv(
        input_dir / "validation-predictions-template.csv",
        "validation-predictions-template.csv",
    )
    december_raw = read_required_csv(
        input_dir / "december-chart-inputs.csv", "december-chart-inputs.csv"
    )
    validate_input_schemas(train_raw, validation_raw, template, december_raw)

    train_prepared = clean_and_engineer(train_raw)
    lower_rpm, upper_rpm = derive_final_rpm_bounds(train_prepared)
    historical_rpm = train_prepared[TARGET] / train_prepared["distance"]
    anomalous_label = ~historical_rpm.between(lower_rpm, upper_rpm)
    train_final = train_prepared.loc[~anomalous_label].copy()

    print("=" * 78)
    print("FINAL ENGINEERED-RPM TRAINING AND INFERENCE")
    print("=" * 78)
    print(f"Rate-per-mile bounds : ${lower_rpm:.4f} to ${upper_rpm:.4f}")
    print(f"Rows used for fit    : {len(train_final):,} of {len(train_prepared):,}")
    print(f"Historical exclusions: {int(anomalous_label.sum()):,}")

    model = build_model()
    model.fit(
        prepare_model_x(train_final),
        train_final[TARGET] / train_final["distance"],
        cat_features=CATEGORICAL_FEATURES,
    )

    validation_prepared = clean_and_engineer(validation_raw)
    predicted_rpm = model.predict(prepare_model_x(validation_prepared))
    validation_pred = enforce_positive(
        predicted_rpm * validation_prepared["distance"].to_numpy(dtype=float),
        "validation",
    )

    predicted_by_id = pd.DataFrame(
        {
            "load_id": validation_raw["load_id"].astype(str),
            "predicted_rate": validation_pred,
        }
    )
    if predicted_by_id["load_id"].duplicated().any():
        raise ValueError("validation.csv contains duplicate load_id values.")

    completed = (
        template[["load_id"]]
        .astype({"load_id": str})
        .merge(predicted_by_id, on="load_id", how="left", validate="one_to_one")
    )
    if completed["predicted_rate"].isna().any():
        raise ValueError("At least one template ID did not receive a prediction.")
    if len(completed) != 12_000:
        raise ValueError(f"Expected 12,000 predictions; found {len(completed):,}.")

    validation_output = output_dir / "validation_predictions.csv"
    completed["predicted_rate"] = completed["predicted_rate"].map(format_rate)
    completed[["load_id", "predicted_rate"]].to_csv(validation_output, index=False)

    daily_dec_market = build_daily_december_market_index(validation_raw)
    december_inference = december_raw.copy()
    december_inference[DATE_COL] = pd.to_datetime(
        december_inference[DATE_COL], errors="coerce"
    )
    december_inference["market_index"] = (
        december_inference[DATE_COL].map(daily_dec_market).astype(float)
    )
    if december_inference["market_index"].isna().any():
        raise ValueError("At least one December row did not receive market_index.")

    december_prepared = clean_and_engineer(december_inference)
    december_rpm = model.predict(prepare_model_x(december_prepared))
    december_pred = enforce_positive(
        december_rpm * december_prepared["distance"].to_numpy(dtype=float),
        "December",
    )

    december_output_frame = december_raw.copy()
    december_output_frame["predicted_rate"] = [format_rate(x) for x in december_pred]
    december_output_frame = december_output_frame[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ]
    december_output = output_dir / "december_predictions.csv"
    december_output_frame.to_csv(december_output, index=False)

    print(f"Created {validation_output} ({len(completed):,} rows)")
    print(f"Created {december_output} ({len(december_output_frame):,} rows)")
    print("predicted_rate is stored with exactly four decimal places.")


if __name__ == "__main__":
    main()
