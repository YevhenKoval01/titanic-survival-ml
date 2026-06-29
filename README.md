# Titanic Survival ML

A machine learning project that explores the Titanic passenger dataset and compares three classification models for survival prediction: Random Forest, Decision Tree, and Logistic Regression.

## Tech Stack

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn

## Repository Name

Recommended GitHub repository name: `titanic-survival-ml`

## Project Structure

```text
data/
  .gitkeep
outputs/
  .gitkeep
reports/
  report.pdf
eda.py
train_models.py
requirements.txt
README.md
```

## Dataset

Place the Titanic dataset in:

```text
data/titanic.csv
```

The dataset is ignored by Git, so the repository stays clean if the data license does not allow redistribution.

## How To Run

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add `titanic.csv` to the `data/` folder.
5. Run the exploratory analysis:

   ```bash
   python eda.py
   ```

6. Run model training and evaluation:

   ```bash
   python train_models.py
   ```

Generated charts are saved in the `outputs/` folder.

