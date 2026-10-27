# Reliability improvements

## Preprocessing

Inference accepts the 28 PCA features (`V1`–`V28`) and either:

- `Amount`: the original, unscaled transaction amount. The fitted training scaler is applied once.
- `scaled_amount`: a value that is already standardized. It is passed through without scaling again.

Do not pass an already-scaled value under the `Amount` column. Keep the feature schema consistent with the data used to train the model.

## Threshold selection and evaluation

Training now uses stratified train, validation, and test partitions. The scaler and SMOTE are fitted/applied only to the training partition. The decision threshold is selected by maximizing F1 on validation probabilities, then evaluated on the held-out test set. The final test set is not used for threshold selection.

The metrics include precision, recall, ROC-AUC, average precision/AUPRC, Matthews correlation coefficient, confusion matrix, and the precision-recall curve arrays. The selected threshold is printed by `train_model.py` and returned as `pipeline["threshold"]`.

The best F1 threshold is a starting point, not a universal operational optimum. For real fraud operations, choose a threshold based on the cost of missed fraud, false alerts, and the review team's capacity.

## Batch predictions

In Streamlit's Batch CSV Upload mode, run predictions and use **Download all predictions (CSV)** to save every input row with `Fraud_Prediction`, `Fraud_Probability`, `Decision_Threshold`, and `Fraud_Label`.

## Run

Install dependencies from `requirements.txt`, train using your local credit-card CSV, then start the app:

```bash
python train_model.py --data-path path/to/creditcard.csv
streamlit run app.py
pytest -q
```

The dataset is not included in the repository; use a legitimate copy of the credit-card fraud dataset and do not commit sensitive transaction data or generated model artifacts.
