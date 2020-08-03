import yaml
import json
import pickle
from os.path import join
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.metrics import mean_absolute_error


params = yaml.safe_load(open('params.yaml'))['train']
random_state: int = params['random_state']
tune: bool = params['tune']
ensemble: bool = params['ensemble']


def remove_outliers(X, y):
    outliers = {
        'LotFrontage': 300,
        'LotArea': 100000,
        'MasVnrArea': 1300,
        'BsmtFinSF1': 5000,
        'BsmtFinSF2': 1200,
        '1stFlrSF': 4000,
        'GrLivArea': 5000,
        'BedroomAbvGr': 7,
        'EnclosedPorch': 500,
        'MiscVal': 6000
    }

    for col, lim in outliers.items():
        if col in X.columns:
            X = X[(X[col] <= lim) | (X[col].isna())]

    y = y[X.index]
    y = y[y < 700000]
    X = X.loc[y.index]

    return X, y


def get_feature_types():
    numerical_features = []
    ordinal_features = []
    categorical_features = []
    with open(join(prepared_data_dir, 'feature_types.txt'), 'r') as f:
        for line in f.readlines():
            feature_name, feature_type = line[:-1].split()
            if feature_type == 'num':
                numerical_features.append(feature_name)
            elif feature_type == 'ord':
                ordinal_features.append(feature_name)
            elif feature_type == 'cat':
                categorical_features.append(feature_name)
            else:
                raise Exception(f'{feature_type} is a wrong feature type for {feature_name}')

    assert len(numerical_features) + len(ordinal_features) + \
        len(categorical_features) == X.shape[1], 'Not all features are present'

    return numerical_features, ordinal_features, categorical_features


def regressor_pipeline(X,
                       y,
                       percentile=80,
                       iterations=1000):
    feature_selector = SelectPercentile(mutual_info_regression,
                                        percentile=percentile)

    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler()),
            ('feature_selection', feature_selector)
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )

    regressor = CatBoostRegressor(cat_features=list(range(len(categorical_features))),
                                  iterations=iterations,
                                  #learning_rate=0.15,
                                  random_state=random_state,
                                  verbose=0)

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ]
    )

    return pipeline


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    prepared_data_dir = join(project_dir, 'data', 'prepared')
    path_to_model_file = join(project_dir, 'models', 'baseline_model.pkl')
    path_to_scores_file = join(project_dir, 'scores.json')

    df_train = pd.read_csv(join(prepared_data_dir, 'train.csv'), index_col=0)

    X, y = df_train.drop(['SalePrice'], axis=1), df_train.SalePrice

    X, y = remove_outliers(X, y)

    numerical_features, ordinal_features, categorical_features = get_feature_types()

    model = regressor_pipeline(X, y, percentile=86, iterations=750)
    model.fit(X, y)

    with open(path_to_model_file, 'wb') as f:
        pickle.dump(model, f)

    df_val = pd.read_csv(join(prepared_data_dir, 'val.csv'), index_col=0)
    X_val, y_val = df_val.drop(['SalePrice'], axis=1), df_val.SalePrice
    mae = mean_absolute_error(y_val, model.predict(X_val))

    print(mae)

    with open(path_to_scores_file, 'w') as f:
        json.dump({'mae': mae}, f)
