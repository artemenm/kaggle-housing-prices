import yaml
import json
import pickle
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

import optuna


params = yaml.safe_load(open('params.yaml'))['train']
random_state: int = params['random_state']
mode: str = params['mode']  # ('', 'ensemble', 'tune')


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


def regressor_pipeline(percentile,
                       iterations,
                       random_state):
    feature_selector = SelectPercentile(mutual_info_regression,
                                        percentile=percentile)

    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
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
                                  loss_function='MAE',
                                  random_state=random_state,
                                  verbose=0)

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ]
    )

    return pipeline


def objective(trial):
    params = {
        'percentile': trial.suggest_int('percentile', 10, 100),
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'random_state': random_state
    }

    model = regressor_pipeline(**params)
    model.fit(X, y)

    # mae = mean_absolute_error(y_val, model.predict(X_val))
    mae = np.mean(cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error', verbose=1))

    print(mae)
    return mae


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    prepared_data_dir = join(project_dir, 'data', 'prepared')
    ensembled_models_dir = join(project_dir, 'models', 'ensembled')
    path_to_model_file = join(project_dir, 'models', 'optuna_model.pkl')
    path_to_scores_file = join(project_dir, 'scores.json')

    df_train = pd.read_csv(join(prepared_data_dir, 'train.csv'), index_col=0)
    X, y = df_train.drop(['SalePrice'], axis=1), df_train.SalePrice

    df_val = pd.read_csv(join(prepared_data_dir, 'val.csv'), index_col=0)
    X_val, y_val = df_val.drop(['SalePrice'], axis=1), df_val.SalePrice

    X, y = remove_outliers(X, y)

    numerical_features, ordinal_features, categorical_features = get_feature_types()

    ##########################
    # TRAIN, SAVE & VALIDATE #
    ##########################

    if mode == 'ensemble':
        models = []
        for i in tqdm(range(25)):
            model = regressor_pipeline(percentile=100,
                                       iterations=1600,
                                       random_state=None)

            model.fit(X, y)
            models.append(model)

            with open(join(ensembled_models_dir, f'model_{i}.pkl'), 'wb') as f:
                pickle.dump(model, f)

        predictions = np.array([model.predict(X_val) for model in models])
        y_pred = np.sum(predictions, axis=0) / predictions.shape[0]
        mae = mean_absolute_error(y_val, y_pred)
    else:
        if mode == 'tune':
            study = optuna.create_study(direction='maximize')
            study.optimize(objective,
                           n_trials=100,
                           n_jobs=-1)
            model = regressor_pipeline(**study.best_params)
        else:
            model = regressor_pipeline(percentile=80,
                                       iterations=1600,
                                       random_state=random_state)

        model.fit(X, y)

        with open(path_to_model_file, 'wb') as f:
            pickle.dump(model, f)

        mae = mean_absolute_error(y_val, model.predict(X_val))

    with open(path_to_scores_file, 'w') as f:
        json.dump({'val_mae': mae}, f)

    print(mae)