# House Prices: sklearn pipeline baseline

A baseline regression model for the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset (Ames, Iowa). I wrote the whole pipeline from a blank file as part of a coding recall routine, so the point here is a clean workflow on messy data, not a leaderboard score.

## Data

`train.csv` from the competition, 1460 houses and 79 features. It's not included in this repo, download it from the Data tab on Kaggle and put it in a `data/` folder.

## What's in the pipeline

Everything that learns from data lives inside one sklearn `Pipeline`, so cross-validation can't leak validation rows into the imputers or encoder.

- Target is `log1p(SalePrice)`, since prices are right skewed
- Split before any fitting, test set untouched until the end
- Three column groups, each with its own mini pipeline:
  - **none_cols** (pool, garage, basement, alley, fence and similar): NaN means the house doesn't have the feature, so it's filled with a constant `'None'` and one-hot encoded
  - **cat_cols** (other categoricals): most_frequent imputer, then one-hot encoder
  - **num_cols**: median imputer, then scaler
- Two models compared with 5-fold CV: linear regression and random forest

## Results

R2 on the log price.

| Model | CV mean | CV std | Test |
|---|---|---|---|
| Linear regression | 0.783 | 0.054 | 0.859 |
| Random forest | 0.859 | 0.030 | 0.883 |

Log-transforming the target did the most for linear regression: 0.696 to 0.783, and the fold-to-fold spread dropped from 0.114 to 0.054. The CV mean is the number to trust for comparing models, the test score is a single split and a sanity check.

## Things I learned

- A lot of the NaNs here aren't missing data. In columns like `PoolQC` or `GarageType`, NaN means "none", and filling with the mean or most common value would invent a pool or a garage.
- A stale fitted pipeline in a notebook gave me a negative R2 once (model trained on dollars, scored against log prices). Restart and run all before trusting a number.

## Next

- Ridge with a few alpha values
- `min_frequency` on the OneHotEncoder to lump rare categories
- Feature engineering (total square footage, house age at sale)
- Gradient boosting

## Run it

Python 3 with pandas, numpy, scikit-learn and jupyter installed. Put `train.csv` in `data/`, open the notebook, and run all cells.
