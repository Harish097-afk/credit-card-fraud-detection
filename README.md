# Credit Card Fraud Detection

A machine-learning project for identifying potentially fraudulent credit-card transactions. It combines imbalanced-data training, model comparison, threshold selection, evaluation metrics, and a Streamlit interface for single-transaction and batch predictions.

> **Project status:** This repository contains the training pipeline, application, automated tests, and evaluation utilities. Performance figures below are explicitly labelled as notebook-reported experiments. Re-run the current pipeline with recorded data, split, seed, and package versions before describing results as independently reproduced or production-ready.

## Contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model performance and comparison](#model-performance-and-comparison)
- [Interactive charts](#interactive-charts)
- [Research-paper comparison and proposed improvements](#research-paper-comparison-and-proposed-improvements)
- [Capacity-based review prioritization](#capacity-based-review-prioritization)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Train and evaluate](#train-and-evaluate)
- [Run the Streamlit application](#run-the-streamlit-application)
- [Tests and CI](#tests-and-ci)
- [Limitations and future research](#limitations-and-future-research)
- [References](#references)

## Problem statement

Credit-card fraud detection is a binary classification problem:

- **Legitimate transaction (`Class = 0`)** — a transaction not labelled as fraud.
- **Fraudulent transaction (`Class = 1`)** — a transaction labelled as fraud.

Fraud is rare compared with legitimate activity. A classifier can therefore achieve high overall accuracy by predicting almost every transaction as legitimate while still failing to detect fraud. This project emphasizes metrics that expose this trade-off, especially **recall, precision, Average Precision (AP/AUPRC), ROC-AUC, and false-positive/false-negative counts**.

## Dataset

This project uses the public [MLG-ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **Documented size:** 284,807 transactions and 31 columns.
- **Target:** `Class`, where 0 is legitimate and 1 is fraud.
- **Features:** `V1`–`V28`, `Time`, `Amount`, and `Class`.
- **Anonymized features:** `V1`–`V28` are transformed numerical features; their original business meanings are not supplied.
- **Time:** elapsed seconds relative to the first transaction in the dataset.
- **Amount:** transaction amount.

The dataset contains 492 fraud cases (about 0.173%), making class imbalance severe. The main training pipeline drops `Time` and uses `V1`–`V28` and `Amount` as model features.

Download `creditcard.csv` from Kaggle and keep it local. **Do not commit the raw dataset or trained model artifacts to GitHub.**

## Methodology

The current repository pipeline follows this process:

1. **Validate data:** required columns, numeric features, finite values, and binary target labels are checked.
2. **Stratified split:** data is divided into training, validation, and held-out test partitions (60% / 20% / 20% overall).
3. **Scale Amount:** fit `StandardScaler` on training data only; reuse it for validation, test, and inference.
4. **Address imbalance:** apply SMOTE to the training partition only. Validation and test data keep their original class distribution.
5. **Train an ensemble:** soft voting combines Random Forest, LightGBM, and XGBoost.
6. **Select threshold:** choose a decision threshold using validation probabilities, not the held-out test labels.
7. **Evaluate:** report precision, recall, ROC-AUC, Average Precision, MCC, and confusion-matrix counts.
8. **Serve predictions:** the Streamlit app supports single transactions and batch CSV files.
9. **Prioritize review:** a separate policy can select the highest-risk percentage of rows for manual review.

### Why the split and SMOTE order matter

SMOTE must be applied **after** the split and only to training data. Oversampling before splitting can let synthetic or related samples influence evaluation and make the result overly optimistic. Keeping validation and test distributions unchanged better reflects the rare-fraud setting.

### Threshold versus review capacity

A **probability threshold** flags every row whose predicted probability is at least the chosen cutoff. **Review capacity** instead ranks a batch and selects the top-risk percentage. They are different policies: capacity controls workload and does not retrain or improve the underlying model.

## Model performance and comparison

The table below reports the project's supplied notebook experiment. The values are **not guaranteed to be reproduced by the current training script**; hardware, split, model configuration, and evaluation procedure must match for an exact comparison.

| Model | Average Precision (AP) ↑ | ROC-AUC ↑ | Reported prediction time ↓ |
|---|---:|---:|---:|
| **Proposed soft-voting ensemble** | **0.8514** | 0.9825 | 1.4253 s |
| Random Forest | 0.8378 | **0.9826** | 0.7830 s |
| XGBoost | 0.8027 | 0.9782 | **0.2038 s** |
| LightGBM | 0.7792 | 0.9757 | 0.2954 s |

**Interpretation**

- The ensemble has the highest reported AP (0.8514), indicating the strongest precision–recall ranking of these four in this experiment.
- Random Forest has a marginally higher ROC-AUC (0.9826 versus 0.9825), so the ensemble does **not** win every metric.
- XGBoost has the lowest reported prediction time in this measurement. Latency depends on hardware, batch size, and measurement setup.
- For an extremely imbalanced dataset, AP/precision–recall analysis and false-positive counts are important complements to ROC-AUC. Accuracy alone can be misleading.

### Interactive charts

**[Open the interactive model comparison chart source](docs/model-comparison.html)**. It includes selectable Plotly charts for AP/ROC-AUC comparison, reported prediction time, and review-capacity trade-offs. GitHub's README renderer does not execute JavaScript embedded in Markdown; to interact with the standalone chart, download `docs/model-comparison.html` from the repository and open it in a browser with an internet connection.

## Research-paper comparison and proposed improvements

### Reference paper

Khalid, Owoh, Uthmani, Ashawa, Osamor, and Adejoh (2024), *Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach*, **Big Data and Cognitive Computing, 8(1), 6**. [Read the paper](https://www.mdpi.com/2504-2289/8/1/6) · [DOI](https://doi.org/10.3390/bdcc8010006).

The paper studies an ensemble involving **SVM, KNN, Random Forest, Bagging, and Boosting**, and discusses both random under-sampling and SMOTE for class imbalance. Its reported evaluation uses its own dataset and experimental setup. The paper's scores should not be directly compared with this project's scores as if they came from the same test set.

### What this project does differently

| Topic | Reference paper's described approach | This project's approach / extension |
|---|---|---|
| Learners | SVM, KNN, RF, Bagging, Boosting and proposed combinations | Soft voting of Random Forest, LightGBM and XGBoost |
| Imbalance handling | Evaluates under-sampling and SMOTE variants | SMOTE is restricted to training partition; test distribution remains imbalanced |
| Evaluation split | Paper describes an 80/20 train/test split for its experiment | Stratified train/validation/test split; validation supports threshold selection |
| Decision threshold | Reports classifier metrics at its experimental decisions | Explicit validation-selected threshold, separated from final test evaluation |
| Operational review | Primarily classifier-level evaluation | Additional top-risk review-capacity policy for a fixed investigation workload |
| Input robustness | Research-method focus | Numeric/finite-value, schema, probability, threshold and capacity validation; automated tests and CI |

These are **methodological and application extensions**, not proof that this project is universally better than the paper. The dataset, split protocol, class prevalence, metrics, and hardware differ. A fair claim of improvement requires reproducing both approaches on the same untouched test data with the same preprocessing and evaluation protocol.

### Research improvements worth testing next

1. **Compare resampling strategies:** training-only SMOTE vs. class-weighted models vs. SMOTE-ENN; preserve the original validation/test distribution.
2. **Use a cost- or capacity-aware threshold:** choose the threshold on validation data using an explicit false-negative/false-positive cost or investigator budget, rather than relying only on F1.
3. **Probability calibration:** evaluate reliability curves and Brier score; calibration matters if risk scores are interpreted as probabilities.
4. **Temporal validation and drift:** because transactions are time-dependent, train on earlier records and evaluate on later records; monitor performance as fraud patterns change.
5. **Explainability:** investigate SHAP or permutation importance on validation data. Since `V1`–`V28` are anonymized, their explanations may not map directly to business concepts.
6. **Efficiency benchmarking:** record training time, inference latency, memory, and throughput under a fixed hardware and batch-size protocol.
7. **Statistical reliability:** repeat experiments with multiple seeds and report variability; avoid selecting a final method by a single split.
8. **Ablation study:** compare individual models, soft voting, sampling strategy, threshold strategy, and review capacity separately to establish which component adds value.

These are proposed experiments. They should not be described as completed results until run and recorded.

## Capacity-based review prioritization

The supplied notebook experiment reported the following results when transactions were ranked by predicted risk and the top percentage was sent for review:

| Review capacity | Transactions reviewed | Fraud caught | False alarms | Fraud missed | Recall | Precision |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 285 | 88 | 197 | 11 | 88.89% | 30.88% |
| 1% | 570 | 90 | 480 | 9 | 90.91% | 15.79% |
| 2% | 1,140 | 91 | 1,049 | 8 | 91.92% | 7.98% |
| 5% | 2,849 | 93 | 2,756 | 6 | 93.94% | 3.26% |
| 10% | 5,697 | 95 | 5,602 | 4 | 95.96% | 1.67% |

The reported test set had 99 frauds. Higher review capacity caught more fraud but also produced many more false alarms and lower precision. These values are from a notebook experiment and should be re-run before being treated as a reproducible benchmark.

**Priority Review does not mean confirmed fraud.** It means a transaction has been selected for investigation. Capacity is a percentage of rows, not a probability threshold, and it does not alter the model itself.

## Repository structure

```text
credit-card-fraud-detection/
├── app.py
├── fraud_detection.py
├── review_policy.py
├── train_model.py
├── compare_models.py
├── requirements.txt
├── tests/
├── docs/
│   └── model-comparison.html
└── DATASET_EVALUATION.md
```

## Setup

Python 3.11 is used by the repository CI workflow. From the repository folder, install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, `py -m pip install -r requirements.txt` may work if the `python` command is not available.

## Train and evaluate

Place the downloaded dataset at `creditcard.csv`, then run:

```bash
python train_model.py --data-path creditcard.csv
python compare_models.py --data-path creditcard.csv
```

The training script saves the ensemble, scaler, selected threshold, and `evaluation_report.json` by default. The comparison script evaluates Logistic Regression, Random Forest, LightGBM, XGBoost, and the soft-voting ensemble. Keep generated artifacts and the dataset out of version control.

## Run the Streamlit application

After training the artifacts:

```bash
streamlit run app.py
```

The app supports:
- **Single transaction:** enter `V1`–`V28` and transaction amount to receive a predicted class and risk score.
- **Batch CSV:** upload rows with `V1`–`V28` and either raw `Amount` or pre-scaled `scaled_amount`.
- **Probability threshold:** flag all rows meeting the selected probability cutoff.
- **Top-risk review capacity:** prioritize the highest-risk 0.5%, 1%, 2%, 5%, or 10% of uploaded rows.
- **CSV export:** download batch results with probability and review-priority fields.

If raw `Amount` is supplied, the saved scaler transforms it. If `scaled_amount` is supplied, it is treated as already standardized. Do not pass both representations unless you have a deliberate reason and understand the app's input handling.

Artifact locations can be configured with `FRAUD_MODEL_PATH`, `FRAUD_SCALER_PATH`, and `FRAUD_THRESHOLD_PATH`.

## Tests and CI

Run the tests locally:

```bash
python -m pytest -q
```

The GitHub Actions workflow runs tests on pushes and pull requests. The tests cover review-capacity selection, malformed inputs, data validation, threshold handling, artifact behavior, and app smoke behavior. Passing software tests establishes code behavior—not real-world fraud performance.

## Limitations and future research

- **Anonymized features:** meanings of `V1`–`V28` are not available.
- **Class imbalance:** high ROC-AUC can coexist with low precision and substantial investigation workload.
- **Temporal generalization:** the dataset covers a short period; random splits may not reflect future transaction streams.
- **Threshold and calibration:** validation-selected thresholds and risk probabilities should be reassessed for deployment conditions.
- **Concept drift:** fraud behavior can change over time.
- **Operational controls:** production use requires security, privacy, monitoring, latency testing, human review, and incident handling.

This is an educational research project, not a certified fraud-prevention service. Do not use it for real financial decisions without independent validation and appropriate controls.

## References

1. Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). **Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach.** *Big Data and Cognitive Computing, 8*(1), 6. [Full text](https://www.mdpi.com/2504-2289/8/1/6) · [DOI: 10.3390/bdcc8010006](https://doi.org/10.3390/bdcc8010006)
2. MLG-ULB. **Credit Card Fraud Detection dataset.** [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). **SMOTE: Synthetic Minority Over-sampling Technique.** *Journal of Artificial Intelligence Research, 16*, 321–357. [Paper](https://doi.org/10.1613/jair.953)
4. [imbalanced-learn — SMOTE documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
5. [scikit-learn — Model evaluation and classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
6. [Streamlit documentation](https://docs.streamlit.io/)
