from __future__ import annotations

import argparse
import time

import pandas as pd
from autogluon.tabular.models import CatBoostModel

from src.Apply_and_Test.Apply_FE import execute_feature_engineering
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns
from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data
from src.utils.get_matrix import get_matrix_core_columns

import warnings

warnings.filterwarnings('ignore')

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


def add_method_metadata(result_matrix, dataset_metadata, X_predict, y_predict, method):
    if method == "pandas":
        result_matrix = add_pandas_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "tabpfn":
        result_matrix = add_tabpfn_metadata_columns(X_predict, y_predict, result_matrix)
    return result_matrix


def feature_addition(X_train, y_train, X_test, y_test, model, method, dataset_metadata, dataset_id, number_of_features):
    if method == "pandas":
        result_matrix = pd.read_parquet("src/Metadata/pandas/Pandas_Matrix_Complete.parquet")
    elif method == "tabpfn":
        result_matrix = pd.read_parquet("src/Metadata/tabpfn/TabFPN_Matrix_Complete.parquet")
    else:
        raise ValueError(f"Method {method} not supported in this simplified script.")

    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    print("Datasets in " + str(method) + " Matrix: " + str(datasets))

    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]

    start = time.time()
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
    comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X_train, y_train, method)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))

    start = time.time()
    X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, method, number_of_features)
    end = time.time()
    print("Time for Predicting Improvement using CatBoost: " + str(end - start))
    try:
        y_list = y_new['target'].tolist()
        y_series = pd.Series(y_list)
        y_new = y_series
    except KeyError:
        pass

    data = concat_data(X_new, y_new, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_best.parquet")
    return X_new, y_new, X_test, y_test


def predict_improvement(result_matrix, comparison_result_matrix, category_or_method, number_of_features):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)

    clf = CatBoostModel()
    clf.fit(X=result_matrix, y=y_result)

    # Predict and score
    comparison_result_matrix.columns = comparison_result_matrix.columns.astype(str)
    comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
    prediction = clf.predict(X=comparison_result_matrix)
    prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
    prediction_concat_df = pd.concat(
        [comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
    best_operation = prediction_concat_df.nlargest(n=number_of_features, columns="predicted_improvement", keep="first")
    best_operation = best_operation.sort_values(key=lambda s: s.str.startswith("without - "), by="feature - name")
    X, y, _, _ = execute_feature_engineering(best_operation)
    return X, y


def process_method(dataset_id, method, model, number_of_features):
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    X_train, y_train, X_test, y_test = feature_addition(X_train, y_train, X_test, y_test, model, method,
                                                        dataset_metadata, dataset_id, number_of_features)
    end = time.time()
    print("Total Time SM: " + str(end - start))
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_best.parquet")


def main(dataset_id, method, number_of_features):
    print("Method: " + str(method) + ", Dataset: " + str(dataset_id) + ", Model: " + str("CatBoost"))
    model = "LightGBM_BAG_L1"

    print(f"\n=== Starting Method: {method} ===")
    process_method(dataset_id, method, model, number_of_features)


def main_wrapper():
    parser = argparse.ArgumentParser(description='Run Surrogate Model with Metadata from Method locally')
    parser.add_argument('--dataset', required=True, help='Dataset ID')
    args = parser.parse_args()

    # We only process 'pandas' now to drop overhead
    methods = ["pandas"]
    number_of_features = 200

    for method in methods:
        main(int(args.dataset), method, number_of_features)


if __name__ == '__main__':
    main_wrapper()