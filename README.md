# Freight Rate Prediction

Machine-learning solution for predicting posted freight rates for 12,000 future loads. The final pipeline uses chronological validation, robust historical-label cleaning, engineered freight features, and `CatBoostRegressor`.

## Final model

The final model predicts **rate per mile (RPM)** and converts it back to dollars:

```text
predicted_rate = predicted_rpm × distance
```

It uses only features available at prediction time:

- pickup, delivery, equipment, and route interactions;
- distance, weight, nonlinear distance transformations, and weight per mile;
- calendar and cyclical time features;
- `market_index` and its interaction with distance;
- flags for missing market information and corrected negative weights.

The model does not use `posted_rate` or any target-derived value as an input feature. Historical rate per mile is used only as the training target and to identify suspicious historical labels.

## Validation strategy and results

The model was evaluated with expanding chronological holdouts: only records before each validation month were used for training. Robust label-cleaning thresholds were recalculated from each fold's training data only.

| Holdout | MAE | RMSE | R² | MAE on normal-label rows |
|---|---:|---:|---:|---:|
| August | $91.25 | $612.89 | 0.8273 | $41.27 |
| September | $109.47 | $616.81 | 0.8360 | $58.41 |
| October | $110.00 | $645.21 | 0.8218 | $49.55 |
| **Mean** | **$103.58** | **$624.97** | **0.8284** | **$49.74** |

These are development-period validation results. Spotter calculates the final hidden-set score after submission.

## Repository structure

```text
freight-rate-prediction/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- score.py
|
|-- src/
|   |-- eda.py
|   |-- validate_model.py
|   `-- train_predict.py
|
|-- data/
|   `-- README.md
|
|-- outputs/
|   |-- validation_predictions.csv
|   `-- december_predictions.csv
|
`-- scorer_results/
    `-- candidate_december.png
```

The raw assessment datasets are intentionally excluded from the public repository. See [`data/README.md`](data/README.md) for setup instructions.

## Setup

Python 3.11 or newer is recommended.

### Windows PowerShell

Check that the Python launcher is installed:

```powershell
py --version
```

Create a virtual environment and install the dependencies. Activation is optional; using the environment's Python executable directly avoids PowerShell script execution-policy issues.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you prefer to activate the environment, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either continue using the direct `.venv` Python commands above or allow scripts only for the current terminal session, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the four supplied CSVs in `data/`, then run commands from the repository root.

On Windows, run the project commands below with `python` after activation. To run without activation, invoke the project virtual environment Python executable directly as shown above. On macOS/Linux, run them after activating `.venv`.

## Reproduce the project

Run the exploratory analysis:

```bash
python src/eda.py
```

Validate the final model chronologically:

```bash
python src/validate_model.py
```

For a faster execution check:

```bash
python src/validate_model.py --quick
```

Train the final model and generate both output files:

```bash
python src/train_predict.py
```

This creates:

- `outputs/validation_predictions.csv`
- `outputs/december_predictions.csv`

Every `predicted_rate` is written as a numeric value with exactly four decimal places. Currency symbols are intentionally excluded because the official scorer requires a numeric column.

Run the supplied official scorer:

```bash
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv
```

The scorer validates 12,000 final predictions and 31 December predictions, then creates `scorer_results/candidate_december.png`.

## Leakage safeguards

- Each validation month is predicted using only earlier months.
- Historical label-quality bounds are learned from each fold's training rows only.
- The final validation dataset is used only for inference.
- December market context is derived from the `market_index` values already provided for the same dates in `validation.csv`; no hidden target is used.

## Author

Melanie Estrada
