import yaml
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np


params = yaml.safe_load(open('params.yaml'))['prepare']
random_state: int = params['random_state']
train_frac: float = params['train_frac']

def fix_features(df):
    # fix the categorical feature that is wrongly considered numerical
    df.MSSubClass = df.MSSubClass.astype('object')

    # sometimes NA is just a value of a categorical feature
    NA_is_not_nan_columns = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]

    for col in NA_is_not_nan_columns:
        df[col] = df[col].replace(np.nan, 'NA')

    # remove correlated features
    corr_features = ['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageCars']
    df = df.drop(corr_features, axis=1)

    return df


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = join(project_dir, 'data', 'raw')
    prepared_data_dir = join(project_dir, 'data', 'prepared')

    df = pd.read_csv(join(raw_data_dir, 'train.csv'), index_col=0)
    df_test = pd.read_csv(join(raw_data_dir, 'test.csv'), index_col=0)

    #####################
    # FIX SOME FEATURES #
    #####################

    df = fix_features(df)
    df_test = fix_features(df_test)

    #################################
    # SAVE NEW TRAIN/VAL/TEST FILES #
    #################################

    df_train = df.sample(frac=train_frac, random_state=random_state)
    df_val = df[~df.index.isin(df_train.index)]

    assert df_train.shape[0] + df_val.shape[0] == df.shape[0], \
        'Incorrect train/val split'

    df_train.to_csv(join(prepared_data_dir, 'train.csv'))
    df_val.to_csv(join(prepared_data_dir, 'val.csv'))
    df_test.to_csv(join(prepared_data_dir, 'test.csv'))

    #################
    # FEATURE TYPES #
    #################

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = list(filter(lambda x: x != 'SalePrice', numerical_features))

    ordinal_features = []

    categorical_features = df.select_dtypes(include=['object']).columns

    with open(join(prepared_data_dir, 'feature_types.txt'), 'w') as f:
        f.write('\n'.join([f'{f} num' for f in numerical_features]) + '\n')
        f.write('\n'.join([f'{f} cat' for f in categorical_features]) + '\n')
