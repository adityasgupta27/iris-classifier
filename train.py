import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load data
df = pd.read_csv("iris.csv")


# Clean dataset
df = df.drop(columns=["Unnamed: 0"])

df = df.rename(columns={
    "0": "sepal_length",
    "1": "sepal_width",
    "2": "petal_length",
    "3": "petal_width",
    "4": "species"
})


# Separate features and target
X = df.drop("species", axis=1)
y = df["species"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Models
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])


# Train
logistic_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)


# Predictions
logistic_predictions = logistic_pipeline.predict(X_test)
svm_predictions = svm_pipeline.predict(X_test)


# Evaluation
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print(classification_report(y_test, logistic_predictions))

print("\nSVM")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

from sklearn.model_selection import cross_val_score
# Cross-validation
logistic_scores = cross_val_score(
    logistic_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

svm_scores = cross_val_score(
    svm_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation results")

print("Logistic Regression:")
print("Scores:", logistic_scores)
print("Mean accuracy:", logistic_scores.mean())

print("\nSVM:")
print("Scores:", svm_scores)
print("Mean accuracy:", svm_scores.mean())

from sklearn.metrics import confusion_matrix

# Confusion matrix
svm_cm = confusion_matrix(y_test, svm_predictions)

print("\nSVM Confusion Matrix:")
print(svm_cm)

import joblib

# Train final model on all available data
svm_pipeline.fit(X, y)

# Save the complete pipeline
joblib.dump(svm_pipeline, "iris_model.joblib")

print("\nFinal SVM model saved as iris_model.joblib")