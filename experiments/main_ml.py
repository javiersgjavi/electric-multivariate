import json
import itertools
import time
import os
import numpy as np
import pandas as pd

from metrics import evaluate
from preprocessing import denormalize
from models import create_model_ml


def read_data(dataset, normalization_method, past_history_factor):
    # read normalization params
    norm_params = None
    with open(
            "../data/{}/{}/norm_params.json".format(dataset, normalization_method), "r",
    ) as read_file:
        norm_params = json.load(read_file)

    # read training / validation data
    tmp_data_path = '../data/{}/{}/{}/'.format(dataset, normalization_method, past_history_factor)

    x_train = np.load(tmp_data_path + "x_train.np.npy")
    y_train = np.load(tmp_data_path + "y_train.np.npy")
    x_test = np.load(tmp_data_path + "x_test.np.npy")
    y_test = np.load(tmp_data_path + "y_test.np.npy")
    y_test_denorm = np.load(tmp_data_path + "y_test_denorm.np.npy")

    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)

    return x_train, y_train, x_test, y_test, y_test_denorm, norm_params


def read_results_file(csv_filepath, metrics):
    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        results = pd.DataFrame(
            columns=[
                "DATASET",
                "MODEL",
                "MODEL_INDEX",
                "MODEL_DESCRIPTION",
                "FORECAST_HORIZON",
                "PAST_HISTORY_FACTOR",
                "PAST_HISTORY",
                "BATCH_SIZE",
                "EPOCHS",
                "STEPS",
                "OPTIMIZER",
                "LEARNING_RATE",
                "NORMALIZATION",
                "TEST_TIME",
                "TRAINING_TIME",
                *metrics,
                "LOSS",
                "VAL_LOSS",
            ]
        )
    return results


def train_trees(model_name, iter_params, x_train, y_train, x_test, norm_params, normalization_method):
    model = create_model_ml(model_name, iter_params)

    x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    print('x_train: {} -> {}'.format(x_train.shape, x_train2.shape))
    training_time_0 = time.time()
    model.fit(x_train2, y_train)
    training_time = time.time() - training_time_0

    x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    print('x_test: {} -> {}'.format(x_test.shape, x_test2.shape))
    test_time_0 = time.time()
    test_forecast = model.predict(x_test2)
    test_time = time.time() - test_time_0

    for i in range(test_forecast.shape[0]):
        nparams = norm_params[0]
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )

    return test_forecast, training_time, test_time


def main_ml(models_ml, datasets, metrics, results_path):
    TRAIN_ML = {
        'tree': train_trees,
        'rf':  train_trees
    }

    for dataset in datasets:
        dataset = dataset.split('/')[-1]

        for model_name in models_ml:

            with open('./parameters.json', "r") as params_file:
                parameters = json.load(params_file)

            for normalization_method, past_history_factor in itertools.product(
                    parameters['normalization_method'],
                    parameters['past_history_factor']
            ):
                csv_filepath = '{}/{}/results.csv'.format(results_path, dataset)
                results = read_results_file(csv_filepath, metrics)

                x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(dataset, normalization_method,
                                                                                         past_history_factor)

                past_history = x_test.shape[1]
                forecast_horizon = y_test.shape[1]

                parameters_models = parameters['model_params'][model_name]

                list_parameters_models = []
                for parameter_field in parameters_models.keys():
                    list_parameters_models.append(parameters_models[parameter_field])

                model_id = 0
                for iter_params in itertools.product(*list_parameters_models):

                    test_forecast, training_time, test_time = TRAIN_ML[model_name](
                        model_name,
                        iter_params,
                        x_train,
                        y_train,
                        x_test,
                        norm_params,
                        normalization_method
                    )

                    if metrics:
                        test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
                    else:
                        test_metrics = {}

                    prediction_path = '{}/{}/{}/{}/{}/{}/'.format(
                        results_path,
                        dataset,
                        normalization_method,
                        str(past_history_factor),
                        'ML',
                        model_name,
                    )

                    if not os.path.exists(prediction_path):
                        os.makedirs(prediction_path)

                    np.save(prediction_path + str(model_id) + '.npy', test_forecast)

                    results = results.append(
                        {
                            "DATASET": dataset,
                            "MODEL": model_name,
                            "MODEL_INDEX": model_id,
                            "MODEL_DESCRIPTION": str(iter_params),
                            "FORECAST_HORIZON": forecast_horizon,
                            "PAST_HISTORY_FACTOR": past_history_factor,
                            "PAST_HISTORY": past_history,
                            "BATCH_SIZE": '',
                            "EPOCHS": '',
                            "STEPS": '',
                            "OPTIMIZER": "Adam",
                            "LEARNING_RATE": '',
                            "NORMALIZATION": normalization_method,
                            "TEST_TIME": test_time,
                            "TRAINING_TIME": training_time,
                            **test_metrics,
                            "LOSS": '',
                            "VAL_LOSS": '',
                        },
                        ignore_index=True
                    )

                    print('END OF EXPERIMENT -> ./results/{}/{}/{}/{}/{}.npy'.format(
                        dataset,
                        normalization_method,
                        past_history_factor,
                        model_name,
                        model_id
                    ))
                    model_id += 1

                    results.to_csv(csv_filepath, sep=";")


if __name__ == '__main__':
    models_ml = ['tree', 'rf']
    datasets = ['./data/cuarentena']
    metrics = ['mse', 'rmse', 'mae', 'mase', 'wape']
    main_ml(models_ml, datasets, metrics, '../results')
