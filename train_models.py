from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


DATA_PATHS = [Path("data/titanic.csv"), Path("titanic.csv")]
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42


def resolve_dataset_path() -> Path:
    for path in DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place titanic.csv in the data/ folder before running this script."
    )


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def impute_age(row: pd.Series, data: pd.DataFrame) -> float:
    if pd.notnull(row["age"]):
        return row["age"]

    title = row["Title"]
    parch = float(row["parch"]) if pd.notnull(row["parch"]) else 0
    sex = row["sex"]

    if title == "Miss" and parch == 0:
        return data[(data["Title"] == "Miss") & (data["parch"] == 0)]["age"].mean()
    if title == "Master":
        return data[data["Title"] == "Master"]["age"].mean()
    if title in ["Sir", "Mr", "Ms", "Mrs", "Dr"]:
        return data[data["Title"] == title]["age"].mean()
    return data[data["sex"] == sex]["age"].mean()


OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(resolve_dataset_path())
df.replace("?", np.nan, inplace=True)

for col in ["age", "fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["survived", "pclass", "embarked", "sex"]:
    df[col] = df[col].astype("category")

df["fare"] = df["fare"].astype(float)
df["Title"] = df["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

df["age"] = df.apply(lambda row: impute_age(row, df), axis=1)
df["fare"] = df.groupby("pclass", observed=False)["fare"].transform(lambda x: x.fillna(x.median()))
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

df["parch"] = pd.to_numeric(df["parch"], errors="coerce").fillna(0)
df["sibsp"] = pd.to_numeric(df["sibsp"], errors="coerce").fillna(0)
df["Family.Size"] = df["parch"] + df["sibsp"] + 1

df["pclass_num"] = pd.to_numeric(df["pclass"])
df.loc[df["age"] > 67, "age"] = 67
fare_cap = df["fare"].quantile(0.99)
df.loc[df["fare"] > fare_cap, "fare"] = fare_cap
df["MPC"] = df["age"] * df["pclass_num"]

features_to_use = ["sex", "pclass", "fare", "Family.Size", "MPC", "Title", "embarked"]
target = "survived"

X = df[features_to_use].copy()
y = df[target].astype(int)

X = pd.get_dummies(X, columns=["sex", "pclass", "Title", "embarked"], drop_first=True)

num_features_to_scale = ["fare", "Family.Size", "MPC"]
scaler = MinMaxScaler()
X[num_features_to_scale] = scaler.fit_transform(X[num_features_to_scale])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=4,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        random_state=RANDOM_STATE,
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}

print("=" * 50)
print("PREDICTIVE MODEL COMPARISON")
print("=" * 50)

roc_data = {}
accuracy_scores = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

    print(f"\n--- Model: {name} ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not survive", "Survived"])
    disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
    axes[i].set_title(f"{name}\nAccuracy: {acc * 100:.2f}%")
    axes[i].grid(False)

plt.suptitle("Confusion matrices for tested models", fontsize=16)
save_plot("comparison_confusion_matrices.png")

plt.figure(figsize=(10, 8))
colors = ["blue", "green", "orange"]
for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), colors):
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve comparison")
plt.legend(loc="lower right")
save_plot("comparison_roc_curves.png")

plt.figure(figsize=(8, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), hue=list(accuracy_scores.keys()), palette="viridis", legend=False)
for i, v in enumerate(accuracy_scores.values()):
    plt.text(i, v + 0.01, f"{v * 100:.2f}%", ha="center", fontweight="bold")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Model accuracy comparison")
save_plot("comparison_accuracy.png")

rf_model = models["Random Forest"]
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, hue=feature_importances.index, palette="viridis", legend=False)
plt.title("Top 10 feature importances - Random Forest")
plt.xlabel("Feature importance")
save_plot("rf_feature_importance.png")

plt.figure(figsize=(24, 12))
plot_tree(
    models["Decision Tree"],
    feature_names=X.columns.tolist(),
    class_names=["Did not survive", "Survived"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10,
)
plt.title("Decision Tree structure - first 3 levels", fontsize=16)
save_plot("decision_tree_structure.png")

coefs = pd.Series(models["Logistic Regression"].coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(12, 8))
sns.barplot(x=coefs.values, y=coefs.index, hue=coefs.index, palette="RdBu", legend=False)
plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
plt.title("Logistic Regression coefficients", fontsize=14)
plt.xlabel("Coefficient weight: negative decreases survival odds, positive increases them")
plt.ylabel("Feature")
save_plot("logistic_regression_coefficients.png")

plt.figure(figsize=(8, 8))
CalibrationDisplay.from_estimator(models["Logistic Regression"], X_test, y_test, n_bins=10, name="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.title("Logistic Regression calibration curve")
plt.legend()
save_plot("logistic_regression_calibration.png")

fig, ax = plt.subplots(figsize=(14, 6))
PartialDependenceDisplay.from_estimator(models["Random Forest"], X_train, ["fare", "MPC"], ax=ax, grid_resolution=50)
plt.suptitle("Random Forest partial dependence plots", fontsize=14)
save_plot("rf_partial_dependence.png")

print(f"\nAll model comparison charts were saved in: {OUTPUT_DIR.resolve()}")
