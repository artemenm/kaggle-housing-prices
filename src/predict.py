import yaml
import pickle
from os.path import join
from pathlib import Path
import pandas as pd


params = yaml.safe_load(open('params.yaml'))['predict']
model_name: str = params['model_name']

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = join(project_dir, 'data', 'raw')
    models_dir = join(project_dir, 'models')

    df_test = pd.read_csv(join(raw_data_dir, 'test.csv'), index_col=0)

    with open(join(models_dir, model_name + '.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(df_test)
    df_test['SalePrice'] = y_pred
    df_test[['SalePrice']].to_csv(join(project_dir, 'submission.csv'))
