# House Prices: sklearn pipeline baseline

A baseline regression model for the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset (Ames, Iowa). I wrote the whole pipeline from a blank file as part of a coding recall routine, so the point here is a clean workflow on messy data and honest model comparison, not a leaderboard score.

## Data

`train.csv` from the competition, 1460 houses and 79 features. It's not included in this repo. Download it from the Data tab on Kaggle and put it in a `data/` folder.

## What's in the pipeline

Everything that learns from data lives inside one sklearn `Pipeline`, so cross-validation can't leak validation rows into the imputers or encoder.

- Target is `log1p(SalePrice)`, since prices are right skewed
- Split before any fitting, test set untouched until the end
- Three column groups, each with its own mini pipeline:
  - **none_cols** (pool, garage, basement, alley, fence and similar): NaN means the house doesn't have the feature, so it's filled with a constant `'None'` and one-hot encoded
  - **cat_cols** (other categoricals): most_frequent imputer, then one-hot encoder
  - **num_cols**: median imputer, then scaler
- Models compared with 5-fold CV on the training set: linear regression, Ridge, random forest

## Results

R2 on the log price, 5-fold CV on the training set. CV is the number I trust for comparing models.

| Model | CV mean | CV std |
|---|---|---|
| Linear regression | 0.783 | 0.054 |
| Ridge (alpha 30) | 0.859 | 0.048 |
| Random forest | 0.859 | 0.030 |

Log-transforming the target did the most for plain linear regression: 0.696 to 0.783, and the fold-to-fold spread dropped from 0.114 to 0.054.

### Ridge: tuning alpha

Alpha is the penalty on large coefficients. Plain linear regression was overfitting the 200+ one-hot columns, many of them rare categories, and the penalty fixed most of that.

| alpha | CV mean | CV std |
|---|---|---|
| 3 | 0.853 | 0.040 |
| 10 | 0.858 | 0.044 |
| 30 | 0.859 | 0.048 |
| 100 | 0.855 | 0.052 |
| 300 | 0.848 | 0.054 |

The top of the curve is flat. Alpha 10, 30 and 100 differ by less than the fold-to-fold spread, so they are effectively tied. I went with 30 because it has the highest mean. Ridge closes the whole gap to the forest, but the forest is still steadier across folds (0.030 vs 0.048).

### Lumping rare categories (`min_frequency`)

`OneHotEncoder(min_frequency=N)` groups any category with fewer than N training rows into one shared "infrequent" column.

| Model | min_frequency | CV mean | CV std |
|---|---|---|---|
| Linear | none | 0.783 | 0.054 |
| Linear | 10 | 0.813 | 0.047 |
| Linear | 70 | 0.821 | 0.055 |
| Ridge (alpha 30) | none or 10 | 0.859 | 0.048 |
| Ridge (alpha 30) | 100 | 0.847 | 0.053 |

- For plain linear regression, lumping helps by about 0.03 and then plateaus. From 70 up the score stays flat, around 0.81 to 0.82.
- For Ridge, lumping at 10 changes nothing and lumping at 100 hurts. The penalty was already shrinking the weights on rare columns, so there was little left to clean up, and a high threshold starts merging categories that carry real price information.
- Even the best lumped linear model (about 0.82) is below unlumped Ridge (0.859). The penalty works better than the data cleaning.
- The gain for linear is about half a fold-to-fold spread, so it's suggestive, not conclusive.

## Things I learned

- A lot of the NaNs here aren't missing data. In columns like `PoolQC` or `GarageType`, NaN means "none", and filling with the mean or most common value would invent a pool or a garage.
- A stale fitted pipeline in a notebook gave me a negative R2 once (model trained on dollars, scored against log prices). Restart and run all before trusting a number.
- The held-out test score moved a few points between runs of nearly identical setups. It's one split of 292 houses, so I compare models on CV and will score the test set once, for the final model.

## Next

- Feature engineering: total square footage, house age at sale, total bathrooms, `MSSubClass` as a category
- Gradient boosting

## Run it

Python 3 with pandas, numpy, scikit-learn and jupyter installed. Put `train.csv` in `data/`, open the notebook, and run all cells.
