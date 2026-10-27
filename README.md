# Credit Card Fraud Detection

A machine-learning project for identifying potentially fraudulent credit-card transactions. It combines imbalanced-data training, model comparison, threshold selection, evaluation metrics, and a Streamlit interface for single-transaction and batch predictions.

> **Project status:** This repository contains the training pipeline, automated tests, and interactive model-results page. Metrics below are reported from project notebook experiments unless explicitly stated otherwise. They are not a guarantee of future or production performance.

## Contents
- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model performance](#model-performance)
- [Interactive charts](#interactive-charts)
- [Review-capacity experiment](#review-capacity-experiment)
- [Research paper and proposed improvements](#research-paper-and-proposed-improvements)
- [Repository structure](#repository-structure)
- [Setup and usage](#setup-and-usage)
- [Tests](#tests)
- [Limitations](#limitations)
- [References](#references)

## Problem statement

Credit-card fraud detection is a binary classification problem:
- **Legitimate transaction (`Class = 0`)** — a transaction not labelled as fraud.
- **Fraudulent transaction (`Class = 1`)** — a transaction labelled as fraud.

Fraud is rare compared with legitimate activity. A classifier can therefore achieve high overall accuracy by predicting almost every transaction as legitimate while still failing to detect fraud. This project emphasizes **Average Precision (AP/AUPRC), ROC-AUC, precision, recall, and false-positive/false-negative counts** rather than accuracy alone.

## Dataset

The project uses the public [MLG-ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- 284,807 transactions and 31 columns.
- 492 labelled frauds (about 0.173%).
- Target: `Class` (0 = legitimate, 1 = fraud).
- `V1`–`V28` are anonymized PCA-transformed features; their original business meanings are not available.
- `Time` is elapsed time from the first transaction; `Amount` is transaction amount.

Download `creditcard.csv` from Kaggle and keep it local. **Do not commit the raw dataset or trained model artifacts to GitHub.**

## Methodology

1. **Validate inputs:** verify expected columns, numeric finite features, and binary labels.
2. **Stratified data split:** create training, validation, and held-out test partitions.
3. **Scale Amount:** fit `StandardScaler` on training data only and reuse it for validation, test, and inference.
4. **Address class imbalance:** apply SMOTE to the training partition only. Validation and test data retain the original class distribution.
5. **Train a soft-voting ensemble:** Random Forest + LightGBM + XGBoost.
6. **Choose a decision threshold:** select it using validation predictions (currently maximizing F1), rather than tuning on the held-out test set.
7. **Evaluate:** report precision, recall, ROC-AUC, Average Precision, MCC, and confusion-matrix counts.
8. **Serve predictions:** Streamlit supports single-transaction analysis and batch CSV scoring.
9. **Prioritize manual review:** batch mode can select the highest-risk percentage of transactions to fit a review capacity.

A probability threshold and review capacity are different policies. A threshold flags all rows above a score cutoff; capacity mode ranks rows and selects a fixed top percentage. A “Priority Review” result is not confirmation of fraud.

## Model performance

Reported test-set model comparison from the project notebook:

| Model | Average Precision (AP) | ROC-AUC | Reported prediction time |
|---|---:|---:|---:|
| **Proposed soft-voting ensemble** | **0.8514** | 0.9825 | 1.4253 s |
| Random Forest | 0.8378 | **0.9826** | 0.7830 s |
| XGBoost | 0.8027 | 0.9782 | **0.2038 s** |
| LightGBM | 0.7792 | 0.9757 | 0.2954 s |

**Interpretation:** the ensemble has the highest reported Average Precision among these four models, while Random Forest has a marginally higher ROC-AUC and XGBoost has the lowest reported prediction time. The best model depends on the trade-off between fraud ranking quality, false alerts, recall, and latency. The timing values are hardware- and measurement-dependent.

These values are from a notebook experiment and should not be represented as independently reproduced by every repository script. For a publication-quality comparison, rerun all models on identical splits and record random seeds, package versions, and measurement conditions.

## Interactive charts

Open the [interactive model comparison](docs/model-comparison.html) in a browser. It includes:
- AP and ROC-AUC comparison across the four models.
- Reported prediction-time comparison.
- Recall and precision across top-risk review capacities.
- A table separating implemented techniques from proposed future research.

The chart uses Plotly loaded from a CDN, so an internet connection is needed to render the interactive plots. The charts visualize reported experiment results; they do not train the models.

## Review-capacity experiment

On the reported test set (56,961 transactions, including 99 frauds), selecting the highest-risk transactions produced:

| Review capacity | Rows reviewed | Frauds caught | False alarms | Missed frauds | Recall | Precision |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 285 | 88 | 197 | 11 | 88.89% | 30.88% |
| 1% | 570 | 90 | 480 | 9 | 90.91% | 15.79% |
| 2% | 1,140 | 91 | 1,049 | 8 | 91.92% | 7.98% |
| 5% | 2,849 | 93 | 2,756 | 6 | 93.94% | 3.26% |
| 10% | 5,697 | 95 | 5,602 | 4 | 95.96% | 1.67% |

The table shows the operational trade-off: increasing review capacity catches more frauds but sends many more legitimate transactions to investigation. It is a ranking experiment on the reported test set, not a probability-threshold result. Validate this policy on fresh or chronological data before operational use.

## Research paper and proposed improvements

### Paper used for research direction

Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). **“Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach.”** *Big Data and Cognitive Computing*, 8(1), 6.

- [Full article (MDPI)](https://www.mdpi.com/2504-2289/8/1/6)
- [DOI link](https://doi.org/10.3390/bdcc8010006)
- [Article XML / methodology and results](https://www.mdpi.com/2504-2289/8/1/6/xml)

The paper discusses under-sampling and SMOTE for class imbalance and evaluates an ensemble approach involving SVM, KNN, Random Forest, bagging, boosting/AdaBoost, and a voting classifier. Its experimental protocol and sampled datasets differ from this repository's reported notebook experiment, so the numbers should **not** be compared as if they came from the same controlled test.

### What this project already does

| Technique | Role in the project | Status |
|---|---|---|
| SMOTE on training data only | Improves minority-class representation without resampling the validation/test sets | Implemented |
| Soft voting (RF + LightGBM + XGBoost) | Combines different tree-based model probabilities | Implemented |
| Validation-selected threshold | Separates threshold tuning from final test evaluation | Implemented |
| Average Precision and ROC-AUC | Evaluate ranking quality under severe class imbalance | Implemented |
| Capacity-based review ranking | Aligns flagged workload with manual review capacity | Implemented |
| Input and artifact validation | Rejects malformed features, invalid probabilities, and invalid thresholds | Implemented and covered by tests |
| CI regression tests | Runs the Python test suite on pushes and pull requests | Configured |

### Research experiments that could improve the proposed method

These are **recommendations to test**, not claims of achieved improvement:

1. **Controlled baseline comparison:** add SVM, KNN, bagging, and AdaBoost baselines inspired by the paper. Compare them against RF, LightGBM, XGBoost, and the voting ensemble using exactly the same stratified splits and metrics.
2. **Cost-sensitive threshold selection:** choose thresholds using explicit costs for missed fraud and false alarms, in addition to the current F1-based baseline.
3. **Probability calibration:** measure calibration (for example, Brier score and reliability curves) before interpreting predicted probabilities as financial risk.
4. **Chronological evaluation:** train on earlier transactions and test on later transactions. The dataset covers roughly two days, so this is only an initial temporal robustness check, not proof of long-term generalization.
5. **Review-policy evaluation:** report precision@k, recall@k, frauds caught per 1,000 reviews, and estimated investigation cost at several capacity limits.
6. **Robustness and drift:** repeat experiments with multiple seeds and monitor performance changes over time.
7. **Explainability:** evaluate permutation importance on validation data or use SHAP carefully. Since `V1`–`V28` are anonymized PCA components, explanations do not directly reveal original transaction attributes.

A defensible research claim requires repeated experiments, a fixed evaluation protocol, confidence intervals where feasible, and a clear distinction between results from this repository and results reported in the cited paper.

## Repository structure

```text
credit-card-fraud-detection/
├── app.py                         # Streamlit prediction UI
├── fraud_detection.py             # Preprocessing, training, evaluation, artifacts
├── review_policy.py               # Capacity-based review selection
├── train_model.py                 # Training and report generation
├── compare_models.py              # Baseline model comparison
├── requirements.txt
├── tests/                         # Regression, validation, and UI tests
├── docs/
│   └── model-comparison.html      # Interactive Plotly charts
└── README.md
```

## Setup and usage

Use Python 3.11 or another Python version supported by the installed dependencies. The repository CI uses Python 3.11.

```bash
git clone https://github.com/Harish097-afk/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in the repository folder.

### Train the ensemble

```bash
python train_model.py --data-path creditcard.csv
```

The script writes model/scaler/threshold artifacts and an evaluation report to its configured output paths. Keep generated artifacts local unless there is a deliberate, secure reason to publish them.

### Compare models

```bash
python compare_models.py --data-path creditcard.csv
```

Use this output for a reproducible comparison; record the split, random seed, package versions, and hardware used for timing.

### Run the application

```bash
streamlit run app.py
```

The application supports single-transaction scoring and batch CSV predictions. Batch input requires `V1`–`V28` and either raw `Amount` or pre-scaled `scaled_amount`. Raw `Amount` is transformed using the saved scaler; do not scale it twice.

Optional artifact path environment variables:
- `FRAUD_MODEL_PATH`
- `FRAUD_SCALER_PATH`
- `FRAUD_THRESHOLD_PATH`

## Neo-brutalist React frontend (optional)

The repository now includes an alternative **React + Vite** frontend with a neo-brutalist visual style: warm cream surfaces, signal-yellow accents, bold borders, offset shadows, responsive layouts, model metrics, transaction analysis, and a batch-review workspace. The original Streamlit interface remains available in `app.py`.

The React frontend calls the FastAPI backend in `api.py`; predictions still come from the existing Python ensemble and saved model/scaler artifacts. The dashboard's benchmark cards show the previously reported notebook experiment metrics and are labelled as reported results, not live measurements.

### 1. Install Python dependencies and train/load artifacts

```bash
python -m pip install -r requirements.txt
python train_model.py --data-path path/to/creditcard.csv
```

If you already have compatible artifacts (`ensemble_model.joblib`, `scaler.joblib`, and optionally `threshold.joblib`), use their configured paths instead. Do not commit the dataset or model artifacts.

### 2. Start the API

From the repository root:

```bash
uvicorn api:app --reload --port 8000
```

API health check: [http://localhost:8000/health](http://localhost:8000/health). Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Start the React frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally [http://localhost:5173](http://localhost:5173)). By default, the frontend connects to `http://localhost:8000`. To use a different API URL, create `frontend/.env` with:

```text
VITE_API_URL=http://localhost:8000
```

Set `FRAUDWATCH_CORS_ORIGINS` on the backend if the frontend is hosted on a different origin. For example, set it to a comma-separated list of explicitly trusted frontend origins.

### 4. Existing Streamlit UI

The original app is still available:

```bash
streamlit run app.py
```

The React interface is an optional frontend, not a replacement for the Python model pipeline. The API requires local model/scaler artifacts before prediction endpoints can return results.

## Tests

```bash
python -m pytest -q
```

The test suite covers review-policy selection, input validation, preprocessing, thresholds, artifact behavior, and app smoke behavior. Passing software tests supports code reliability; it does not prove real-world model performance.

## Limitations

- **Anonymized predictors:** `V1`–`V28` have no available original business descriptions.
- **Extreme class imbalance:** precision and investigation workload must be examined alongside recall.
- **Short time span:** the dataset covers about two days; random splitting alone cannot establish future performance.
- **Synthetic sampling:** SMOTE can create synthetic minority samples that may not fully represent real fraud patterns.
- **Potential distribution shift:** fraud patterns change; monitor performance and recalibrate/retrain only through a controlled process.
- **Operational risk:** a real payment system needs security, latency testing, human review, auditability, monitoring, and independent validation.

This is an educational/research project, not a certified fraud-prevention service. Do not use it to make real financial decisions without rigorous independent validation and appropriate controls.

## References

1. Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). “Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach.” *Big Data and Cognitive Computing*, 8(1), 6. [Article](https://www.mdpi.com/2504-2289/8/1/6) · [DOI](https://doi.org/10.3390/bdcc8010006).
2. MLG-ULB. [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. imbalanced-learn. [SMOTE documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).
4. scikit-learn. [Model evaluation and classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
5. Streamlit. [Documentation](https://docs.streamlit.io/).
