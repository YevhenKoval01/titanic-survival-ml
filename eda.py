from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATHS = [Path("data/titanic.csv"), Path("titanic.csv")]
OUTPUT_DIR = Path("outputs")


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


OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(resolve_dataset_path())
df.replace("?", np.nan, inplace=True)

num_cols = ["age", "fare", "survived", "pclass", "sibsp", "parch"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 5))
sns.histplot(data=df, x="sex", hue="survived", multiple="fill", shrink=0.8, palette="Set2")
plt.title("Survival rate by sex")
plt.ylabel("Proportion")
save_plot("eda_sex_survived.png")

plt.figure(figsize=(6, 5))
sns.histplot(
    data=df,
    x="pclass",
    hue="survived",
    multiple="fill",
    shrink=0.8,
    palette="Set2",
    discrete=True,
)
plt.title("Survival rate by ticket class")
plt.xticks([1, 2, 3])
plt.ylabel("Proportion")
save_plot("eda_pclass_survived.png")

plt.figure(figsize=(6, 5))
df_embarked = df.dropna(subset=["embarked"])
sns.histplot(data=df_embarked, x="embarked", hue="survived", multiple="fill", shrink=0.8, palette="Set2")
plt.title("Survival rate by embarkation port")
plt.ylabel("Proportion")
save_plot("eda_embarked_survived.png")

plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="age", hue="survived", fill=True, common_norm=False, palette="Set2", alpha=0.5)
plt.title("Age distribution by survival status")
save_plot("eda_age_density.png")

plt.figure(figsize=(8, 5))
df_fare_vis = df[df["fare"] < 200]
sns.kdeplot(data=df_fare_vis, x="fare", hue="survived", fill=True, common_norm=False, palette="Set2", alpha=0.5)
plt.title("Fare distribution by survival status")
save_plot("eda_fare_density.png")

plt.figure(figsize=(8, 6))
corr_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation matrix for numerical variables")
save_plot("eda_correlation.png")

print(f"EDA charts were saved in: {OUTPUT_DIR.resolve()}")
