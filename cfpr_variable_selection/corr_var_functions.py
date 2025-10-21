# This file contains all of the utility functions for the correlation-based exogenous variable selection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import math
from datetime import datetime

scaler_mm = MinMaxScaler()
scaler_ss = StandardScaler()


def scale_df(df, scaler, freq):
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    scaled_df.index = pd.date_range(start=df.index[0], freq=freq, periods=len(df))
    return scaled_df


def ss_scale_a_df(df, freq):
    return scale_df(df, scaler_ss, freq)


def mm_scale_a_df(df, freq):
    return scale_df(df, scaler_mm, freq)


def lag_a_df(df, start_date=None, end_date=None, freq="MS", lag_end=25, lag_start=0):
    start_date = start_date or df.index[0]
    end_date = end_date or df.index[-1]

    lagged_dfs = [
        df.shift(-i).rename(columns=lambda x: f"Lagged_{i}_Month")
        for i in range(lag_start, lag_end)
    ]
    lagged_df = pd.concat(lagged_dfs, axis=1)
    lagged_df.index = pd.date_range(start=start_date, end=end_date, freq=freq)
    return lagged_df


def is_monthly_frequency(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        return False

    freq = pd.infer_freq(df.index)
    return freq == "MS"


def filter_monthly_dataframes(dict):
    monthly_dataframes = {
        key: df for key, df in dict.items() if is_monthly_frequency(df)
    }
    return monthly_dataframes


# ---------Cleaning text with regex ---------#
def clean_text(text, patterns):
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    return text


def remove_und(text):
    return clean_text(text, {"_": " "})


def remove_capitals(text):
    return clean_text(text, {"[A-Z]": ""})


def remove_reg(text):
    return clean_text(text, {r"\[.*?\]": ""})


# -------------------------------------------#


def sort_by_date(dict):
    date_info = [
        {
            "File Name": name,
            "Start Date": df.index[0].timestamp(),
            "End Date": df.index[-1].timestamp(),
        }
        for name, df in dict.items()
    ]
    return pd.DataFrame(date_info).sort_values(by="Start Date")


# Selection with hard cutoff, only available data passes through
def exo_date_select(
    dict, target_name, start_date, end_date, log=None, scaler=None, freq="MS"
):
    cut_dfs = {}
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    for key, df in dict.items():
        if isinstance(df.index, pd.DatetimeIndex):
            if (df.index <= start_date).any() and (df.index >= end_date).any():
                cut_dfs[key] = df

    cut_dfs = {
        name: df.reindex(
            pd.date_range(start=start_date, end=end_date, freq=freq)
        ).interpolate()
        for name, df in cut_dfs.items()
    }

    target = cut_dfs.pop(target_name)
    exo = pd.concat(cut_dfs.values(), axis=1)
    exo.columns = cut_dfs.keys()

    log_functions = {"log": np.log, "log10": np.log10, "log2": np.log2}
    if log in log_functions:
        exo, target = log_functions[log](exo), log_functions[log](target)

    exo.replace([math.inf, -math.inf], np.nan, inplace=True)
    exo.interpolate(inplace=True)

    scale_functions = {"ss": ss_scale_a_df, "mm": mm_scale_a_df}
    if scaler in scale_functions:
        exo, target = scale_functions[scaler](exo, freq=freq), scale_functions[scaler](target, freq=freq)

    return target, exo


# Selection with soft cutoff, all data passes through
def exo_date_select_interpolate(
    dict, target_name, start_date, end_date, log=None, scaler="ss", freq="MS"
):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    cut_dfs = {
        name: df.reindex(date_range).interpolate(
            method="linear", limit_direction="both"
        )
        for name, df in dict.items()
    }

    target = cut_dfs.pop(target_name)
    exo = pd.concat(cut_dfs.values(), axis=1)
    exo.columns = cut_dfs.keys()

    exo = exo.apply(pd.to_numeric)
    target = target.apply(pd.to_numeric)

    log_functions = {"log": np.log, "log10": np.log10, "log2": np.log2}
    if log in log_functions:
        exo, target = log_functions[log](exo), log_functions[log](target)

    exo.replace([math.inf, -math.inf], np.nan, inplace=True)
    exo.interpolate(inplace=True)

    scale_functions = {"ss": ss_scale_a_df, "mm": mm_scale_a_df}
    if scaler in scale_functions:
        exo, target = scale_functions[scaler](exo, freq=freq), scale_functions[scaler](target, freq=freq)

    return target, exo


def threshold_correlation(target, exogenous, ignore=None, thresh=0.5, select_unique=1):
    corr = pd.concat([target, exogenous], axis=1).corr().unstack().reset_index()
    corr.columns = ["Target Lag", "Exo Variable", "Corr"]
    corr = corr[
        corr["Target Lag"].str.contains("Lagged")
        & ~corr["Exo Variable"].str.contains("Lagged")
    ]

    if ignore:
        for item in ignore:
            corr = corr[~corr["Exo Variable"].str.contains(item)]

    filtered_corr = corr[abs(corr["Corr"]) >= thresh].sort_values(
        by="Corr", ascending=False
    )
    exo_vars = {
        var: filtered_corr[filtered_corr["Exo Variable"] == var].iloc[:select_unique]
        for var in filtered_corr["Exo Variable"].unique()
    }

    return exo_vars


def full_corr_selection(
    dict,
    target_name,
    start,
    end,
    ignore=None,
    thresh=0.5,
    select_unique=1,
    freq="MS",
    lag_end=25,
    lag_start=0,
    interpolate_cutoff=False,
    log=None,
    scaler="ss",
):
    exo_func = exo_date_select_interpolate if interpolate_cutoff else exo_date_select
    target, exogenous = exo_func(dict, target_name, start, end, log, scaler, freq)

    target_lag = lag_a_df(
        target,
        start_date=start,
        end_date=end,
        freq=freq,
        lag_end=lag_end,
        lag_start=lag_start,
    )
    exo_vars = threshold_correlation(
        target_lag, exogenous, ignore, thresh, select_unique
    )

    corr_selection = pd.concat(exo_vars.values(), axis=0).reset_index(drop=True)

    print(f"\nCorrelation Summary for {target_name}\n")
    print(
        f"Threshold: {thresh}\nLog: {log}\nScaler: {scaler}\nDate Range: {start} to {end}"
    )
    print(
        f"Cutoff Interpolated: {interpolate_cutoff}\nNumber of exogenous regressors considered: {len(list(exogenous.columns))}"
    )
    print(f"Number of exogenous regressors above threshold: {len(exo_vars)}\n")
    print(corr_selection.to_string())

    return corr_selection, list(exo_vars.keys())
