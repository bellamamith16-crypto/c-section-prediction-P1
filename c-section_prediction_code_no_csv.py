# C-Section Prediction
# Machine Learning project using Decision Tree and Random Forest
# Dataset is fetched automatically from the UCI Machine Learning Repository.
#
# No CSV file is required.
#
# Required packages:
#   pip install ucimlrepo pandas numpy matplotlib scikit-learn

import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")


# Install ucimlrepo automatically if it is not available.
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 70)
print("C-SECTION PREDICTION USING MACHINE LEARNING")
print("=" * 70)


# -------------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------------
# UCI Caesarian Section Classification Dataset
# UCI dataset ID: 472
#
# The dataset contains 80 records and the target variable is Caesarian.
dataset = fetch_ucirepo(id=472)

# Use the original table so that the ID column is also retained,
# matching the structure used in the project report.
data = dataset.data.original.copy()

# Clean column names
data.columns = [str(col).strip() for col in data.columns]

# Find the target column safely
target_candidates = [
    col for col in data.columns
    if str(col).strip().lower() in ["caesarian", "caesarean"]
]

if not target_candidates:
    raise ValueError(
        "Target column 'Caesarian' was not found in the UCI dataset."
    )

target_col = target_candidates[0]

print("\nFirst five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# -------------------------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# -------------------------------------------------------------------
X = data.drop(columns=[target_col])
y = data[target_col]

# Convert all feature columns to numeric values.
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# Remove rows containing unexpected missing/non-numeric values.
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

print("\nFeature columns:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# -------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# -------------------------------------------------------------------
# 4. DECISION TREE
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("DECISION TREE")
print("=" * 70)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=0
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print(f"\nDecision Tree test accuracy: {tree_accuracy:.4f}")

print("\nDecision Tree confusion matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nDecision Tree classification report:")
print(classification_report(y_test, y_pred_tree, zero_division=0))


# -------------------------------------------------------------------
# 5. RANDOM FOREST
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    random_state=4
)

random_forest.fit(X_train, y_train)

y_pred_forest = random_forest.predict(X_test)

forest_accuracy = accuracy_score(y_test, y_pred_forest)

print(f"\nRandom Forest test accuracy: {forest_accuracy:.4f}")

print("\nRandom Forest confusion matrix:")
print(confusion_matrix(y_test, y_pred_forest))

print("\nRandom Forest classification report:")
print(classification_report(y_test, y_pred_forest, zero_division=0))


# -------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind="barh")
plt.title("Feature Importances - Random Forest")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# 7. CROSS-VALIDATION
# -------------------------------------------------------------------
# The project report uses 10-fold cross-validation.
# Because this is a small dataset, stratified folds are used.
print("\n" + "=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

cv_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=10,
    scoring="accuracy"
)

print("10-fold cross-validation scores:")
print(np.round(cv_scores, 4))

print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")


# -------------------------------------------------------------------
# 8. FINAL COMPARISON
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Decision Tree Accuracy : {tree_accuracy:.4f}")
print(f"Random Forest Accuracy : {forest_accuracy:.4f}")

if forest_accuracy >= tree_accuracy:
    print("\nRandom Forest performed better or equal on the test set.")
else:
    print("\nDecision Tree performed better on the test set.")

print("\nNote:")
print(
    "This project is for educational/research purposes. "
    "Predictions from this model should not replace clinical judgment "
    "or medical diagnosis."
)
