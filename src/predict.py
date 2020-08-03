import yaml
import pickle
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd


params = yaml.safe_load(open('params.yaml'))['predict']
model_name: str = params['model_name']
mode: str = params['mode']

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = join(project_dir, 'data', 'raw')
    models_dir = join(project_dir, 'models')
    ensembled_models_dir = join(models_dir, 'ensembled')

    df_test = pd.read_csv(join(raw_data_dir, 'test.csv'), index_col=0)

    if mode == 'ensemble':
        models = []
        for i in range(25):
            with open(join(ensembled_models_dir, f'model_{i}.pkl'), 'rb') as f:
                model = pickle.load(f)
                models.append(model)
        predictions = np.array([model.predict(df_test) for model in models])
        y_pred = np.sum(predictions, axis=0) / predictions.shape[0]
    else:
        with open(join(models_dir, model_name + '.pkl'), 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(df_test)

    df_test['SalePrice'] = y_pred
    df_test[['SalePrice']].to_csv(join(project_dir, 'submission.csv'))
