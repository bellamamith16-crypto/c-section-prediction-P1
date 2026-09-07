# C-Section Prediction - P1
# Machine Learning project using Decision Tree and Random Forest
# Dataset source: UCI Caesarian Section Classification Dataset (ID 472)
#
# Install required packages:
# pip install pandas numpy matplotlib scikit-learn ucimlrepo

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from ucimlrepo import fetch_ucirepo


# 1. LOAD DATASET

dataset = fetch_ucirepo(id=472)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Make target a 1-D Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Use clear column names where available
X.columns = ["Age", "Delivery", "Delivery.1", "Blood", "Heart"]

data = X.copy()
data["Caesarian"] = y.values

print("First five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
data.info()

print("\nSummary of dataset:")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

print("\nTarget variable (Caesarian):")
print(data["Caesarian"].value_counts())


# 2. DECISION TREE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDecision Tree:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Decision Tree using Gini index
clf_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=0
)

clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

print(
    "\nDecision Tree test accuracy: "
    f"{accuracy_score(y_test, y_pred_gini):.4f}"
)

y_pred_train_gini = clf_gini.predict(X_train)

print(
    "Decision Tree training accuracy: "
    f"{accuracy_score(y_train, y_pred_train_gini):.4f}"
)

print(
    "Decision Tree training set score: "
    f"{clf_gini.score(X_train, y_train):.4f}"
)

print(
    "Decision Tree test set score: "
    f"{clf_gini.score(X_test, y_test):.4f}"
)

plt.figure(figsize=(12, 8))
tree.plot_tree(
    clf_gini,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gini))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_gini))


# 3. RANDOM FOREST

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X,
    y,
    random_state=42
)

print("\nRandom Forest:")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

clf = RandomForestClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    random_state=4
)

clf.fit(X_train_rf, y_train_rf)

print("\nRandom Forest feature importances:")
print(clf.feature_importances_)

print("\nDataset columns:")
print(data.columns)

y_pred = clf.predict(X_test_rf)

print("\nRandom Forest predictions:")
print(y_pred)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test_rf, y_pred))

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test_rf, y_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test_rf, y_pred))

print("\nRandom Forest 10-fold Cross-validation scores:")
cv_scores = cross_val_score(clf, X_train_rf, y_train_rf, cv=10)
print(cv_scores)


# 4. FEATURE IMPORTANCE VISUALIZATION

features = X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(
    range(len(indices)),
    importances[indices],
    align="center"
)
plt.yticks(
    range(len(indices)),
    [features[i] for i in indices]
)
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()


# END OF PROJECT

print("\nC-Section Prediction project completed.")
