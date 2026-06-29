# House Price Prediction

[Kaggle *House Prices – Advanced Regression Techniques*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

The goal is to predict the final sale price of residential homes in Ames, Iowa from
79 explanatory variables, evaluated on the RMSE of `log(SalePrice)`.

---

## Results

| Model | CV RMSE (log price) |
| --- | --- |
| Linear Regression (baseline) | unstable* |
| Random Forest | 0.142 |
| Gradient Boosting | 0.131 |
| Ridge | 0.131 |
| **Lasso (best)** | **0.130** |

Best model on a held-out validation split: **RMSE ≈ 0.131, R² ≈ 0.91**.

\* The plain linear baseline is numerically unstable on the wide one-hot feature
matrix — a nice illustration of why regularisation (Ridge/Lasso) matters here.

---

## Notebooks

The project is split into four notebooks, meant to be read in order:

| # | Notebook | What it does |
| --- | --- | --- |
| 00 | [`00-EDA.ipynb`](notebooks/00-EDA.ipynb) | Exploratory data analysis: missing values, target distribution, numeric & categorical feature relationships with price |
| 01 | [`01-Feature-Engineering.ipynb`](notebooks/01-Feature-Engineering.ipynb) | Domain-aware imputation, ordinal encoding, engineered features (ages, total area, baths, flags), skew correction, one-hot encoding |
| 02 | [`02-Modeling.ipynb`](notebooks/02-Modeling.ipynb) | Cross-validated comparison of 5 regression models and selection of the best |
| 03 | [`03-Evaluation.ipynb`](notebooks/03-Evaluation.ipynb) | Residual diagnostics, feature importance, and Kaggle submission file |

---

## Project structure

```text
House-Price-Prediction/
├── data/                  # raw CSVs (git-ignored) + generated data/processed/
│   ├── train.csv          # download from Kaggle
│   └── test.csv           # download from Kaggle
├── notebooks/
│   ├── 00-EDA.ipynb
│   ├── 01-Feature-Engineering.ipynb
│   ├── 02-Modeling.ipynb
│   └── 03-Evaluation.ipynb
├── report/
│   ├── data_description.txt   # official feature documentation
│   └── sample_submission.csv
├── requirements.txt
└── README.md
```

---

## Getting started

1. **Clone & install dependencies**

   ```bash
   git clone https://github.com/<your-username>/House-Price-Prediction.git
   cd House-Price-Prediction
   pip install -r requirements.txt
   ```

2. **Download the data** from the
   [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   and place `train.csv` and `test.csv` in the `data/` folder.

3. **Run the notebooks in order** (00 → 01 → 02 → 03):

   ```bash
   jupyter notebook
   ```

   Notebook 01 writes the processed features to `data/processed/`, notebook 02 trains
   and saves the best model, and notebook 03 writes the predictions to
   `report/submission.csv`.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · SciPy · matplotlib · seaborn

---

## Key takeaways

- Most "missing" values in this dataset encode *absence* (no basement, garage, pool …)
  rather than unknown data, so they are imputed with domain knowledge.
- `log1p` transforming the skewed target and skewed numeric features improves the fit.
- Engineered features (total square footage, total bathrooms, property ages) rank among
  the most important predictors, alongside overall quality and living area.
- Regularised linear models (Lasso/Ridge) are both accurate and interpretable on this
  problem, matching Gradient Boosting while staying simple.
