import argparse
from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def main(input_csv: str, output_arrow: str):
    """
    Main function to read a CSV file, exclude 'date' column, and convert
    remaining columns to Arrow format.

    Args:
        input_csv (str): Path to the input CSV file.
        output_arrow (str): Path to save the Arrow file.
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 排除 'date' 列
    time_series_columns = [col for col in df.columns if col.lower() != 'date']
    time_series = [df[col].to_numpy() for col in time_series_columns]

    # 转换为Arrow格式
    output_path = Path(output_arrow)
    convert_to_arrow(output_path, time_series, compression="lz4")


if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Convert a CSV file to Arrow format.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing time series data.",
    )
    parser.add_argument(
        "--output_arrow",
        type=str,
        required=True,
        help="Path to save the output Arrow file.",
    )

    args = parser.parse_args()
    main(args.input_csv, args.output_arrow)

"""
python csv_to_arrow.py \
      --input_csv /home/zhupengtian/zhangqingliang/datasets/electricity.csv \
      --output_arrow /home/zhupengtian/zhangqingliang/datasets/electricity.arrow
"""