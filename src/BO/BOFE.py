from __future__ import annotations

import numpy as np
import sklearn.gaussian_process as gp
import multiprocessing
import time
import pandas as pd
import psutil
from scipy.optimize import minimize
from scipy.stats import norm

from src.Apply_and_Test.Apply_FE import execute_feature_engineering_recursive

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data
from src.utils.get_matrix import get_matrix_core_columns
from multiprocessing import Value
import ctypes

import warnings
warnings.filterwarnings('ignore')

last_reset_time = Value(ctypes.c_double, time.time())

merge_keys = ["dataset - id", "feature - name", "operator", "model", "improvement"]

def safe_merge(left, right):
    return pd.merge(left, right, on=merge_keys, how="inner")


def create_empty_core_matrix_for_dataset(X_train, model, dataset_id) -> pd.DataFrame:
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)
    for feature1 in X_train.columns:
        featurename = "without - " + str(feature1)
        columns = get_matrix_core_columns()
        new_rows = pd.DataFrame(columns=columns)
        operator = "delete"
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurename,
            operator,
            model,
            0
        ]
        comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    columns = get_matrix_core_columns()
    new_rows = pd.DataFrame(columns=columns)
    featurenames = create_featurenames(X_train.columns)
    for i in range(len(featurenames)):
        operator, _ = extract_operation_and_original_features(featurenames[i])
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurenames[i],
            operator,
            model,
            0
        ]
    comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    return comparison_result_matrix


def recursive_feature_addition(X, y, X_test, y_test, model, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id):
    result_matrix = pd.read_parquet("../Metadata/core/Core_Matrix_Complete.parquet")
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()

    # Sample from result_matrix with all transformations and results
    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] == dataset_id]
        result_matrix = result_matrix.sample(n=30, random_state=42)

    comparison_result_matrix = create_empty_core_matrix_for_dataset(X, model, dataset_id)
    # Predict and split again
    start = time.time()
    X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, X, y, wanted_min_relative_improvement)
    end = time.time()
    print("Time for Predicting Improvement using CatBoost: " + str(end - start))
    if X_new.equals(X):  # if X_new.shape == X.shape
        try:
            y_list = y['target'].tolist()
            y_series = pd.Series(y_list)
            y = y_series
        except KeyError:
            print("")
        data = concat_data(X, y, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_BO.parquet")
        return X, y
    else:
        try:
            y_list = y_new['target'].tolist()
            y_series = pd.Series(y_list)
            y_new = y_series
        except KeyError:
            print("")
        data = concat_data(X_new, y_new, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_BO.parquet")
        return recursive_feature_addition(X_new, y_new, X_test, y_test, model, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id)


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def predict_improvement(result_matrix, comparison_result_matrix, X_train, y_train, wanted_min_relative_improvement):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
    comparison_result_matrix.columns = comparison_result_matrix.columns.astype(str)
    comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
    for col in X_train.columns:
        X_train[col], _ = pd.factorize(X_train[col], sort=True)
        X_train[col] = X_train[col].fillna(0).astype(int)
    y_train, _ = pd.factorize(y_train)
    y_train = pd.Series(y_train)
    kernel = gp.kernels.ConstantKernel()
    model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=1, normalize_y=True)
    model.fit(X_train, y_train)
    print("GP fitted")
    random_search = False
    bounds = np.array([])
    n_params = bounds.shape[0]
    if random_search:
        ei = -1 * expected_improvement(result_matrix, model, y_result, greater_is_better=True, n_params=n_params)
        prediction = result_matrix[np.argmax(ei), :]
    else:
        prediction = sample_next_hyperparameter(expected_improvement, model, y_result, result_matrix, greater_is_better=True, bounds=bounds)
    print("Prediction:", str(prediction))
    prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
    prediction_concat_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
    best_operation = prediction_concat_df.nlargest(n=1, columns="predicted_improvement", keep="first")
    if best_operation["predicted_improvement"].values[0] < wanted_min_relative_improvement:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - not good enough")
        return X_train, y_train
    else:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - execute feature engineering")
        X, y, _, _ = execute_feature_engineering_recursive(best_operation, X_train, y_train)
    return X, y

def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, result_matrix, greater_is_better=False, bounds=(0, 10)):
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    res = minimize(fun=acquisition_func,
                   x0=result_matrix["improvement"],
                   bounds=bounds,
                   method='L-BFGS-B',
                   args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
    if res.fun < best_acquisition_value:
        best_acquisition_value = res.fun
        best_x = res.x
    return best_x

def process_method(dataset_id, model, wanted_min_relative_improvement, last_reset_time):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    X_train, y_train = recursive_feature_addition(X_train, y_train, X_test, y_test, model, dataset_metadata, None, wanted_min_relative_improvement, dataset_id)
    end = time.time()
    print("Total Time SM: " + str(end - start))
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_BO.parquet")


def main(dataset_id, wanted_min_relative_improvement):
    print("Dataset: " + str(dataset_id) + ", Model: " + str("CatBoost"))
    model = "LightGBM_BAG_L1"
    process_method(dataset_id, model, wanted_min_relative_improvement, last_reset_time)


def run_with_resource_limits(target_func, mem_limit_mb, time_limit_sec, check_interval=5):
    process = multiprocessing.Process(target=target_func)
    process.start()
    pid = process.pid

    while process.is_alive():
        try:
            mem = psutil.Process(pid).memory_info().rss / (1024 * 1024)  # MB
            elapsed_time = time.time() - last_reset_time.value
            if mem > mem_limit_mb:
                print(f"[Monitor] Memory exceeded: {mem:.2f} MB > {mem_limit_mb} MB. Terminating.")
                process.terminate()
                break

            if elapsed_time > time_limit_sec:
                print(f"[Monitor] Time limit exceeded: {elapsed_time:.1f} sec > {time_limit_sec} sec. Terminating.")
                process.terminate()
                break

        except psutil.NoSuchProcess:
            break
        time.sleep(check_interval)

    process.join()
    return process.exitcode


def main_wrapper():
    #parser = argparse.ArgumentParser(description='Run CatBoost Surrogate Model with Metadata from Method')
    #parser.add_argument('--dataset', required=True, help='Metafeature Method')
    #args = parser.parse_args()
    wanted_min_relative_improvement = 0.1
    main(359993, wanted_min_relative_improvement)


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    main_wrapper()
