export DATA_PATH='input/data/train_regression.csv'
export PROBLEM_TYPE='single_col_regression'
export SHFFLE='True'
export KFOLDS=5
export TARGET_COLS='SalePrice LotArea'
export MULTI_DELIMITER=' '
export SAVE_PATH='input/data/train_single_col_regression_folds.csv'
export HOLDOUT = 'holdout_20'

python ./src/cross_validation.py
