# baseline-classifier

A first, deliberately simple machine-learning model on a public dataset, built with a clean evaluation setup.

## Task
Prepare a public dataset and get a first simple model working, with:
- a train/test split done before anything else
- a trivial baseline to beat
- one accuracy number you trust

## Dataset
Breast Cancer Wisconsin (Diagnostic), a UCI dataset bundled with scikit-learn (569 samples, 30 numeric features, binary target).

## Method
1. Load data and split 80/20, stratified, `random_state=42`. The test set is untouched until the end.
2. Baseline: `DummyClassifier(strategy="most_frequent")`.
3. Model: `StandardScaler` + `LogisticRegression` in one pipeline, so scaling is fit on training data only (no leakage).
4. Evaluate on the test set once.

## Results

| Model | Test accuracy |
|---|---|
| Majority-class baseline | 0.6316 |
| Scaled Logistic Regression | **0.9825** |

The model makes 2 errors on 114 test samples (1 false negative, 1 false positive).

## Run it
```bash
pip install -r requirements.txt
jupyter notebook baseline_model.ipynb
```

## Notes
- The test set is small, so one mistake shifts accuracy by about 0.9 points. Treat 0.98 as approximate.
- Next steps: cross-validation, other models, and recall on the malignant class.
