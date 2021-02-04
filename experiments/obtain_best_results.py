import openpyxl
import os
import shutil
import pandas as pd
import numpy as np


def save_best_results(metric):
    """ It makes a csv with the best model for each dataset in the terms of the selected metric """
    datasets = os.listdir('../results/')
    base = pd.read_csv('../results/' + datasets[0] + '/results.csv', error_bad_lines=False, sep=';',
                       index_col='Unnamed: 0')
    columns = base.columns.values
    models = base['MODEL'].unique()
    result_best_models = pd.DataFrame(columns=columns)

    for dataset in datasets:
        results_dataset = pd.read_csv('../results/' + dataset + '/results.csv', error_bad_lines=False, sep=';',
                                      index_col='Unnamed: 0')
        for model in models:
            minimum = np.min(results_dataset.loc[results_dataset['MODEL'] == model, [metric]]).values[0]
            best_model = results_dataset.loc[(results_dataset[metric] == minimum) &
                                             (results_dataset['MODEL'] == model), :]
            result_best_models = result_best_models.append(best_model)

    if not os.path.exists('../results_best/'):
        os.mkdir('../results_best/')

    if not os.path.exists('../results_best/lists/'):
        os.mkdir('../results_best/lists/')

    result_best_models.to_csv('../results_best/lists/results_best_' + metric + '.csv', sep=';', index=False)
    return result_best_models


def save_comparative_tables(result_best_models, metric):
    """It makes an excel with a comparative table of each model for any dataset and forecast horizon"""
    models = result_best_models['MODEL'].unique()
    datasets = result_best_models['DATASET'].unique()
    forecast_horizons = result_best_models['FORECAST_HORIZON'].unique()

    if not os.path.exists('../results_best/tables/'):
        os.mkdir('../results_best/tables/')

    excel_path = '../results_best/tables/table_best_models_' + metric + '.xlsx'
    excel = pd.ExcelWriter(excel_path, engine='openpyxl')
    excel.book = openpyxl.Workbook()

    for horizon in forecast_horizons:

        res = pd.DataFrame(columns=models)
        for dataset in datasets:
            row = []

            for model in models:
                row.append(result_best_models.loc[(result_best_models['DATASET'] == dataset) &
                                                  (result_best_models['FORECAST_HORIZON'] == horizon) &
                                                  (result_best_models['MODEL'] == model), :][metric].values[0])
            res.loc[dataset, :] = row
        res.to_excel(excel, sheet_name=str(horizon))

    excel.save()
    default_sheet = excel.book[excel.book.sheetnames[0]]
    excel.book.remove(default_sheet)

    excel.close()


def obtain_paths_predictions(result_best_models):
    """It obtains the path of the file .npy, which contains the prediction of the model"""
    paths = []
    models = result_best_models.reset_index()
    for i in range(len(models)):
        model = models.loc[i, :]

        path = model['DATASET'] + '/' + str(model['NORMALIZATION']) + '/' + str(
            model['PAST_HISTORY_FACTOR']) + '/' + str(model['EPOCHS']) + '/' + str(model['BATCH_SIZE']) + '/' + str(
            model['LEARNING_RATE']) + '/' + model['MODEL'] + '/' + str(model['MODEL_INDEX']) + '.npy'

        paths.append(path)
    return paths


def save_best_predictions(paths, metric):
    """It saves the predictions in the dir ./results_best"""
    dir_res = '../results/'
    dir_predictions = '../results_best/predictions/'
    dir_dest = dir_predictions+metric+'/'

    if not os.path.exists(dir_predictions):
        os.mkdir(dir_predictions)

    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest)

    for path in paths:
        dataset = path.split('/')[0].strip()
        modelo = path.split('/')[-2].strip()
        nombre = dataset + '_' + modelo + '.npy'

        if not os.path.exists(dir_dest + dataset):
            os.mkdir(dir_dest + dataset)

        shutil.copyfile(dir_res + path, dir_dest + '/' + dataset + '/' + nombre)



def obtain_best_results():
    metrics = ['wape', 'TRAINING_TIME']
    for metric in metrics:
        result_best_models = save_best_results(metric)
        save_comparative_tables(result_best_models, metric)
        paths = obtain_paths_predictions(result_best_models)
        save_best_predictions(paths, metric)
    print('[INFO] Results of the best models saved in ./results_best. Models evaluated by:', metrics)


if __name__ == '__main__':
    obtain_best_results()
