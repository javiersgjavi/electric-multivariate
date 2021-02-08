import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def read_ts_dataset(filename, delimiter=","):
    """
            Function for reading csv dataset with multiple time series and storing them into an array
            :param delimiter:
            :param filename:
            :return: array of time series
            """
    with open(filename, "r") as datafile:
        data = datafile.readlines()
        data = np.asarray(
            [np.asarray(l.rstrip().split(delimiter), dtype=np.float32) for l in data]
        )
    return data


def normalize(data, norm_params, method="zscore"):
    """
    Normalize time series
    :param data: time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: normalized time series
    """
    assert method in ["zscore", "minmax", None]

    if method == "zscore":
        std = norm_params["std"]
        if std == 0.0:
            std = 1e-10
        return (data - norm_params["mean"]) / norm_params["std"]

    elif method == "minmax":
        denominator = norm_params["max"] - norm_params["min"]

        if denominator == 0.0:
            denominator = 1e-10
        return (data - norm_params["min"]) / denominator

    elif method is None:
        return data


def denormalize(data, norm_params, method="zscore"):
    """
    Reverse normalization time series
    :param data: normalized time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: time series in original scale
    """
    assert method in ["zscore", "minmax", None]

    if method == "zscore":
        return (data * norm_params["std"]) + norm_params["mean"]

    elif method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]

    elif method is None:
        return data


def get_normalization_params(data):
    """
    Obtain parameters for normalization
    :param data: time series
    :return: dict with string keys
    """
    d = data.flatten()
    norm_params = {}
    norm_params["mean"] = d.mean()
    norm_params["std"] = d.std()
    norm_params["max"] = d.max()
    norm_params["min"] = d.min()

    return norm_params


def normalize_dataset(train, test, normalization_method, dtype="float32"):
    # Normalize train data
    norm_params = []
    for i in range(train.shape[0]):
        nparams = get_normalization_params(train[i])
        train[i] = normalize(
            np.array(train[i], dtype=dtype), nparams, method=normalization_method
        )
        norm_params.append(nparams)

    # Normalize test data
    test = np.array(test, dtype=dtype)
    for i in range(test.shape[0]):
        nparams = norm_params[i]
        test[i] = normalize(test[i], nparams, method=normalization_method)

    return train, test, norm_params


def moving_windows_preprocessing(train_ts, test_ts, y, past_history, forecast_horizon, dtype, core=1):
    # Format training and test input/output data using the moving window strategy
    x_train, y_train = [], []
    x_test, y_test = [], []

    for i, ts in tqdm(
            list(enumerate(train_ts[y])), desc="Moving window preprocesing train ({})".format(core)
    ):

        if len(ts) >= past_history + forecast_horizon:
            # Training data
            for j in range(past_history, ts.shape[0] - forecast_horizon + 1, 24):
                indices = list(range(j - past_history, j))

                window_ts = []
                for time_series in train_ts:
                    window_ts.append(time_series.flatten()[indices])
                window = np.array(window_ts)

                x_train.append(window)
                y_train.append(ts[j: j + forecast_horizon])

    for i, ts in tqdm(
            list(enumerate(test_ts[y])), desc="Moving window preprocesing test ({})".format(core)
    ):
        if len(ts) >= past_history + forecast_horizon:
            # Testing data
            for j in range(past_history, ts.shape[0] - forecast_horizon + 1, 24):
                indices = list(range(j - past_history, j))

                window_ts = []
                for time_series in test_ts:
                    window_ts.append(time_series.flatten()[indices])
                window = np.array(window_ts)

                x_test.append(window)
                y_test.append(ts[j: j + forecast_horizon])

    return (
        np.array(x_train).astype(dtype),
        np.array(y_train).astype(dtype),
        np.array(x_test).astype(dtype),
        np.array(y_test).astype(dtype),
    )

