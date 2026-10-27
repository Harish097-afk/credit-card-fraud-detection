# Real Dataset Evaluation and App Smoke Test

The repository's CI tests use synthetic data to verify code paths. They do **not** establish real-world fraud-detection performance.

## 1. Obtain the benchmark data

Use the European cardholder transaction dataset commonly distributed through Kaggle (original ULB dataset). Keep the downloaded CSV private/local; do not commit transaction records. The expected columns are `V1`–`V28`, `Amount`, and `Class` (0 = legitimate, 1 = fraud). The optional `Time` column is ignored.

## 2. Train and evaluate on a held-out test split

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python train_model.py --data-path path/to/creditcard.csv
```

On Windows PowerShell, for example:

```powershell
python train_model.py --data-path "C:\Users\YOUR_USER\Downloads\creditcard.csv"
```

This command:
- reports the dataset size and fraud prevalence;
- uses separate training, validation, and held-out test splits;
- applies SMOTE only to the training split;
- chooses the decision threshold on validation data, not on the test set;
- reports precision, recall, ROC-AUC, average precision/AUPRC, MCC, and the confusion matrix;
- writes `evaluation_report.json` plus model/scaler/threshold artifacts.

Review false negatives (missed fraud) and false positives (legitimate transactions flagged) alongside precision and recall. Because the classes are extremely imbalanced, accuracy alone is not an adequate metric. The validation-selected F1 threshold is only a starting point, not a business-approved operating threshold.

## 3. Smoke-test the Streamlit UI

The CI suite now runs a Streamlit AppTest against temporary, synthetic model artifacts and clicks **Analyze Transaction** to verify that a prediction and probability render. This checks basic app wiring, not browser deployment or real model quality.

Run the full suite with:

```bash
python -m pytest -q
```

To launch the app locally after training:

```bash
streamlit run app.py
```

Then verify both Single Transaction and Batch CSV Upload modes, including the downloaded output CSV. A deployed public URL has not been configured or verified by this repository workflow.
