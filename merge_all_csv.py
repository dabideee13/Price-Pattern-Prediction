#!/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Aggregate Stock Data
"""

import os
import glob
import subprocess
import pandas as pd


if __name__ == '__main__':

    # Set working directory or path to data
    path = "/Users/d.e.magno/Datasets/stocks"

    # Match csv files by pattern
    all_files = glob.glob(os.path.join(path, "*.csv"))

    # Combine all files in the list and export as csv
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv(os.path.join(path, "stocks_merged.csv"))
