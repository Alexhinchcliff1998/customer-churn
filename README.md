# Retention Signals | Customer Churn

Compare logistic regression and random forest using training-only stratified cross-validation and average precision. Evaluate on a held-out synthetic dataset and compare with a dummy baseline.

```bash
pip install -r requirements.txt
python churn.py
python -m pytest -q
```
Generates synthetic_customers.csv, results.json and confusion_matrix.png. Reference run selected logistic regression, with holdout ROC-AUC 0.792 and average precision 0.700; dummy average precision 0.396. No real-world retention impact claimed. Independent AI-assisted demonstration.
