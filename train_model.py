import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = "data/ml_dataset.csv"
MODEL_PATH = "models/xgb_model.pkl"
TEST_SPLIT = 0.8   # 80% past, 20% future
MAX_MISSING_RATIO = 0.8

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Dataset loaded:", df.shape)


DROP_COLS = [
    'date',
    'symbol',
    'report_date',
    'label',
    'future_return'
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df['label']

X = X.replace([np.inf, -np.inf], np.nan)

# Drop columns that are mostly missing
missing_ratio = X.isna().mean()
X = X.loc[:, missing_ratio < MAX_MISSING_RATIO]

print("Features after pruning:", X.shape[1])

# Fill remaining NaNs with median
X = X.fillna(X.median(numeric_only=True))

# -----------------------------
# TIME-BASED TRAIN / TEST SPLIT
# -----------------------------
split_index = int(len(X) * TEST_SPLIT)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -----------------------------
# TRAIN XGBOOST MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, preds))

roc = roc_auc_score(y_test, probs)
print("ROC AUC:", round(roc, 4))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
fi = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 20 Features:")
print(fi.head(20))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
