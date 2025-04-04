# Copyright contributors to the TSFM project
#
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import requests

from tsfm_public.toolkit.time_series_preprocessor import extend_time_series
from tsfm_public.toolkit.util import select_by_index


@pytest.fixture(scope="module")
def ts_data():
    dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

    forecast_length = 96
    context_length = 512
    timestamp_column = "date"

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )

    test_data = select_by_index(
        data,
        start_index=12 * 30 * 24 + 4 * 30 * 24 - context_length * 5,
        end_index=12 * 30 * 24 + 4 * 30 * 24,
    ).reset_index(drop=True)

    test_data["id"] = np.array(["a", "b", "c", "d", "e"]).repeat(context_length)

    return test_data, {
        "timestamp_column": timestamp_column,
        "id_columns": ["id"],
        "target_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "prediction_length": forecast_length,
    }


def get_inference_response(
    msg: Dict[str, Any],
) -> pd.DataFrame:
    URL = "http://127.0.0.1:8000/v1/inference/forecasting"
    headers = {}
    req = requests.post(URL, json=msg, headers=headers)

    #
    if req.ok:
        resp = req.json()

        df = [pd.DataFrame.from_dict(r) for r in resp["results"]]
        return df
    else:
        return req


def encode_data(df: pd.DataFrame, timestamp_column: str) -> Dict[str, Any]:
    df[timestamp_column] = df[timestamp_column].apply(lambda x: x.isoformat())
    data_payload = df.to_dict(orient="list")
    return data_payload


def test_zero_shot_forecast_inference(ts_data):
    test_data, params = ts_data
    id_columns = params["id_columns"]

    prediction_length = 96
    num_ids = test_data[id_columns[0]].nunique()

    # test single
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length

    # test single, more data
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    test_data_ = extend_time_series(
        test_data_, params["timestamp_column"], grouping_columns=id_columns, freq="1h", periods=10
    )
    test_data_ = test_data_.fillna(0)

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length

    # test multi-time series
    test_data_ = test_data.copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)

    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length * num_ids

    # test multi-time series, errors
    test_data_ = test_data.copy()
    test_data_ = test_data_.iloc[3:]

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    out = get_inference_response(msg)
    assert "Received 509 time points for id a" in out.text

    # test multi-time series, multi-id
    # error due to insufficient context
    test_data_ = test_data.copy()
    test_data_ = test_data_.iloc[3:]
    test_data_["id2"] = test_data_[params["id_columns"]]

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"] + ["id2"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    out = get_inference_response(msg)
    assert "Received 509 time points for id ('a', 'a')" in out.text

    # single series, less columns
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"][:4],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length
    assert df_out[0].shape[1] == 6

    # single series, less columns, no id
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": [],
            "target_columns": ["HULL"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length
    assert df_out[0].shape[1] == 2

    # single series, different prediction length
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            "prediction_length": params["prediction_length"] // 4,
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length // 4

    # single series
    # error wrong prediction length
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            "prediction_length": params["prediction_length"] * 4,
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    out = get_inference_response(msg)
    assert "prediction_filter_length should be positive" in out.text

    # single series, different prediction length
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()

    msg = {
        "model_id": "ibm/test-ttm-v1",
        "parameters": {
            "prediction_length": params["prediction_length"] // 4,
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"][1:],
        },
        "data": encode_data(test_data_, params["timestamp_column"]),
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length // 4


@pytest.mark.parametrize(
    "model_path",
    [
        "ibm/test-patchtst",
        "ibm/test-patchtsmixer",
    ],
)
def test_trained_model_inference(ts_data, model_path):
    test_data, params = ts_data
    id_columns = params["id_columns"]

    prediction_length = 96

    # test single
    test_data_ = test_data[test_data[id_columns[0]] == "a"].copy()
    encoded_data = encode_data(test_data_, params["timestamp_column"])

    msg = {
        "model_id": model_path,
        "parameters": {
            # "prediction_length": params["prediction_length"],
        },
        "schema": {
            "timestamp_column": params["timestamp_column"],
            "id_columns": params["id_columns"],
            "target_columns": params["target_columns"],
        },
        "data": encoded_data,
        "future_data": {},
    }

    df_out = get_inference_response(msg)
    assert len(df_out) == 1
    assert df_out[0].shape[0] == prediction_length
