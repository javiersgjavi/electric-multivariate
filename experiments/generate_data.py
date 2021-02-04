# -*- coding: utf-8 -*-
import os
import requests
import json
import time
import random
import itertools
import numpy as np
from tqdm import tqdm
from preprocessing import (
    read_ts_dataset,
    normalize_dataset,
    moving_windows_preprocessing,
    denormalize,
)

NUM_CORES = 7

def notify_slack(msg, webhook=None):
    if webhook is None:
        webhook = os.environ.get("webhook_slack")
    if webhook is not None:
        try:
            requests.post(webhook, json.dumps({"text": msg}))
        except:
            print("Error while notifying slack")
            print(msg)
    else:
        print("NO WEBHOOK FOUND")


# Preprocessing parameters
with open("parameters.json") as f:
  PARAMETERS = json.load(f)


#PARAMETERS = json.load("parameters.json")
NORMALIZATION_METHOD = PARAMETERS["normalization_method"]
PAST_HISTORY_FACTOR = PARAMETERS[
    "past_history_factor"
]  # past_history = forecast_horizon * past_history_factor


# This variable stores the urls of each dataset.
#DATASETS = json.load("../data/datasets.json")
with open("../data/datasets.json") as f:
  DATASETS = json.load(f)

DATASET_NAMES = [d for d in list(DATASETS.keys())]

def generate_dataset(args):
    dataset, norm_method, past_history_factor = args

    train_url = DATASETS[dataset]["train"]
    test_url = DATASETS[dataset]["test"]

    train = read_ts_dataset("../data/{}/train.csv".format(dataset))
    test = read_ts_dataset("../data/{}/test.csv".format(dataset))

    forecast_horizon = 24 #test.shape[1]

    print(
        dataset,
        {
            "Number of time series": train.shape[0],
            "Max length": np.max([ts.shape[0] for ts in train]),
            "Min length": np.min([ts.shape[0] for ts in train]),
            "Forecast Horizon": forecast_horizon,
        },
    )


    # Normalize data
    train, test, norm_params = normalize_dataset(
        train, test, norm_method, dtype="float32"
    )


    norm_params_json = [{k: float(p[k]) for k in p} for p in norm_params]
    norm_params_json = json.dumps(norm_params_json)

    with open("../data/{}/{}/norm_params.json".format(dataset, norm_method), "w") as f:
        f.write(norm_params_json)

    # Format training and test input/output data using the moving window strategy
    past_history = int(forecast_horizon * past_history_factor)


    x_train, y_train, x_test, y_test = moving_windows_preprocessing(
        train, test, past_history, forecast_horizon, np.float32, n_cores=NUM_CORES
    )

    y_test_denorm = np.copy(y_test)
    #i = 0
    for i in range(y_test.shape[0]):
        y_test_denorm[i] = denormalize(y_test[i], norm_params[0], method=norm_method)

    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print()
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)

    np.save(
        "../data/{}/{}/{}/x_train.np".format(dataset, norm_method, past_history_factor),
        x_train,
    )
    np.save(
        "../data/{}/{}/{}/y_train.np".format(dataset, norm_method, past_history_factor),
        y_train,
    )
    np.save(
        "../data/{}/{}/{}/x_test.np".format(dataset, norm_method, past_history_factor),
        x_test,
    )
    np.save(
        "../data/{}/{}/{}/y_test.np".format(dataset, norm_method, past_history_factor),
        y_test,
    )
    np.save(
        "../data/{}/{}/{}/y_test_denorm.np".format(
            dataset, norm_method, past_history_factor
        ),
        y_test_denorm,
    )


params = [
    (dataset, norm_method, past_history_factor)
    for dataset, norm_method, past_history_factor in itertools.product(
        DATASET_NAMES, NORMALIZATION_METHOD, PAST_HISTORY_FACTOR
    )
]

for i, args in tqdm(enumerate(params)):
    t0 = time.time()
    generate_dataset(args)
    dataset, norm_method, past_history_factor = args
    notify_slack(
        "[{}/{}] Generated dataset {} with {} normalization and past history factor of {} ({:.2f} s)".format(
            i, len(params), dataset, norm_method, past_history_factor, time.time() - t0
        )
    )
    print(
        "[{}/{}] Generated dataset {} with {} normalization and past history factor of {} ({:.2f} s)".format(
            i, len(params), dataset, norm_method, past_history_factor, time.time() - t0
        )
    )


