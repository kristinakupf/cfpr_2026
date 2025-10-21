import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import math

scaler_mm = MinMaxScaler()
scaler_ss = StandardScaler()

"""
Function: plot_targets
Plots simple graph of target variable(s)
Takes in the frames of variables, labels for each varibable in the legend, 
colors for the lines, title and axis labels, and settings for showing the legend or grid
"""
def plot_targets(dfs, num_targets=1, legend_labels=None, colors=None, title='', xlabel='', ylabel='', legend_tf=False, grid_tf=False):
    if legend_labels is None:
        legend_labels = [''] * num_targets
    if colors is None:
        colors = [None] * num_targets
    
    for i in range(num_targets):
        plt.plot(dfs[i], label=legend_labels[i], color=colors[i])
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if legend_tf=='True':
        plt.legend()
    if grid_tf=='True':
        plt.grid()
    
    plt.show()


"""
Function: ss_scale_a_df
Scales a given dataframe using scikitlearn's StandardScaler function and 
restores the original datetime frequency and column titles
Takes in a df and the frequency of its datetime index (ie 'MS')
Returns a scaled df
"""
def ss_scale_a_df(df, freq):
    df_s = pd.DataFrame(scaler_ss.fit_transform(df))
    start_date = df.index[0]
    num_rows = len(df)
    date_range = pd.date_range(start=start_date, freq=freq, periods=num_rows)
    df_s.index = date_range
    df_s.columns = df.columns
    return df_s

"""
Function: mm_scale_a_df
Scales a given dataframe using scikitlearn's MinMaxScaler function and 
restores the original datetime frequency and column titles
Takes in a df and the frequency of its datetime index (ie 'MS')
Returns a scaled df
"""
def mm_scale_a_df(df, freq):
    df_s = pd.DataFrame(scaler_mm.fit_transform(df))
    start_date = df.index[0]
    num_rows = len(df)
    date_range = pd.date_range(start=start_date, freq=freq, periods=num_rows)
    df_s.index = date_range
    df_s.columns = df.columns
    return df_s

"""
Function: remove_num
Removes any numbers in a given set of text and return the number-less text
Very useful when removing footnote numbers left behind in columns of stats canada data like so:
df.columns = [remove_num(col) for col in df.columns]
"""
def remove_num(text):
    return re.sub(r'\d+', '', text)

"""
Function: remove_reg
Removes anything inside [] and including the brakcet in a given set of text and return the number-less text
Very useful when removing footnote []s left behind in columns of stats canada data like so:
df.columns = [remove_num(col) for col in df.columns]
"""
def remove_reg(text):
    return re.sub(r'\[.*?\]', '', text)

"""
Removes underscores
"""
def remove_und(text):
    return re.sub(r'_', ' ', text)

"""
Removes capital letters
"""
def remove_capitals(text):
    new_text = re.sub(r'[A-Z]', '', text)
    return new_text

"""
Function: lag_a_df
Takes in a df and creates a 'reverse lag' of the data over the set time provided
Returns a lagged df with te proper datetime index restored and the columns named accordingly
Automatically lags 24 months into the future unless specified using freq and lag_end
"""
def lag_a_df(df, start_date=None, end_date=None, freq='MS', lag_end=25):

    if start_date == None:
        start_date = df.index[0]
    
    if end_date == None:
        end_date = df.index[-1]

    lagging_dfs = []
    for i in range(1,lag_end):
        lag_df = df.shift(i)
        lag_df.columns = [f'Lagged_{i}_Month']
        lagging_dfs.append(lag_df)

    lagged_df = pd.concat(lagging_dfs, axis=1)

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    lagged_df.index = date_range
    return lagged_df

"""
Scales, logs, and lags all in one
"""
def lag_scale_and_log(df, start_date, end_date, scaler='ss', freq='MS', lag_length=25, log=None):
    fdf = df.loc[start_date:end_date]
    fdf = fdf.apply(pd.to_numeric)

    if log == 'log':
        fdf_l = np.log(fdf)
    elif log == 'log2':
        fdf_l = np.log2(fdf)
    elif log == 'log10':
        fdf_l = np.log10(fdf)
    else:
        fdf_l = fdf

    if scaler == 'ss':
        fdf_s = ss_scale_a_df(fdf_l, freq=freq)
    elif scaler == 'mm':
        fdf_s = mm_scale_a_df(fdf_l, freq=freq)
    else:
        fdf_s = fdf_l

    lagged_fdf = lag_a_df(fdf_s, lag_end=lag_length)
    
    return lagged_fdf

"""
Function: threshold_correlation
Takes in a target df, exogenous variable df, a list of any variables to ignore, and a threshold
Returns a df of sorted varibales that correlate above the threshold 
"""
def threshold_correlation(target, exogenous, ignore=None, thresh=0.5):
    conc = pd.concat([target, exogenous], axis=1)
    corr = conc.corr()
    corr_pairs = corr.unstack().reset_index()
    corr_pairs.columns = ['Target Lag', 'Exo Variable', 'Corr']

    #corr_pairs = corr_pairs[corr_pairs['Target Lag'] < corr_pairs['Exo Variable']]
    corr_pairs = corr_pairs[corr_pairs['Target Lag'].str.contains('Lagged')]
    corr_pairs = corr_pairs[~corr_pairs['Exo Variable'].str.contains('Lagged')]

    if ignore==None:
        pass
    else:
        for i in range(len(ignore)):
            corr_pairs = corr_pairs[~corr_pairs['Exo Variable'].str.contains(ignore[i])]

    filtered_corr_pairs = corr_pairs[abs(corr_pairs['Corr']) >= thresh]

    sorted_corr_pairs = filtered_corr_pairs.sort_values(by='Corr', ascending=False)

    return sorted_corr_pairs

"""
Function: variable_extraction
Takes in a sorted correlation table and makes a dictionary of dataframes, 
each dataframe holds a unique variable from the sorted table and its top (5) 
correlation coefficients and corresponding lag times.
Returns the dictionary and a list of the unique variables that were above the threshold
"""
def variable_extraction(sorted_corr, show_num=5):
    vars_list = sorted_corr['Exo Variable'].unique()
    exo_vars = {}

    for i in vars_list:
        target_var = sorted_corr[sorted_corr['Exo Variable'] == i].iloc[:show_num,:]
        exo_vars[i] = target_var

    return exo_vars

"""
Function: sort_by_date
Creates a dataframe from a dictionary that holds the titles of each dataframe in the dict, the start and end dates, and sorts it by date
"""
def sort_by_date(dict):
    file_names = list(dict.keys())
    
    date_frame = pd.DataFrame({'File Name': 'a', 'Start Date': 'a', 'End Date':'a'}, index=[0]) #dummy row to stop future warning
    date_frame = date_frame.reset_index()
    date_frame.drop(columns=date_frame.columns[0], inplace=True)

    for m in range(len(file_names)):
        df = dict[file_names[m]]
        new_row = {'File Name':file_names[m], 'Start Date': df.index[0], 'End Date': df.index[-1]}
        date_frame = pd.concat([date_frame, pd.DataFrame([new_row])], ignore_index=True)

    date_frame.drop(axis=0, index=0, inplace=True)

    date_frame = date_frame.sort_values(by='Start Date')
    return date_frame

"""
Function: exo_date_select
Returns the target and the exogenous variable dataframe of dataframes filtered to only include dfs within the hard date range defined
Only inner NaN interpolation in this, if a df is not within the range, it is not used
Also returns the dictionary with all of the cut dataframes (not scaled or logged)
Any values that create a -inf or inf log result will get replaced with NaN and interpolated before feeding into ss
"""
def exo_date_select(dict, target_name, start_date, end_date, log=None, scaler='ss', freq='MS'):
    date_frame = sort_by_date(dict)
    filtered_date_frame = date_frame.loc[(date_frame['Start Date'] <= start_date) & (date_frame['End Date'] >= end_date)]
    filtered_names = list(filtered_date_frame['File Name']) #the dfs that fit in the specified range without the need for interpolation

    cut_dfs = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    for n in range(len(filtered_names)):
        cut_dfs[filtered_names[n]] = pd.DataFrame(dict[filtered_names[n]])
        cut_dfs[filtered_names[n]] = cut_dfs[filtered_names[n]].reindex(date_range)
        cut_dfs[filtered_names[n]] = cut_dfs[filtered_names[n]].interpolate(method='linear')

    target = cut_dfs[target_name]
    cut_dfs.pop(target_name, None)
    exo_names = list(cut_dfs.keys())

    exo = pd.concat(cut_dfs.values(), axis=1)
    exo.index = date_range
    exo.columns = exo_names

    if log == 'log':
        exo = np.log(exo)
        target = np.log(target)
    elif log == 'log10':
        exo = np.log10(exo)
        target = np.log10(target)
    elif log == 'log2':
        exo = np.log2(exo)
        target = np.log2(target)
    else:
        pass   

    # Replace problematic values with a nan and then interpolate
    exo_clean = exo.copy()
    exo_clean[exo_clean == -math.inf] = np.nan
    exo_clean = exo_clean.interpolate(method='linear', limit_direction='forward')

    if scaler == 'ss':
        exo_s = ss_scale_a_df(exo, freq=freq)
        target_s = ss_scale_a_df(target, freq=freq)
        
    elif scaler == 'mm':
        exo_s = mm_scale_a_df(exo, freq=freq)
        target_s = mm_scale_a_df(target, freq=freq)
    else:
        exo_s = exo
        target_s = target

    return target_s, exo_s, cut_dfs

"""
Function: exo_date_select_interpolate
Returns the (scaled and/or logged) target and the exogenous variable dataframe made from of dataframes within the date range defined, 
where all dfs out of range are interpolated to become valid.
Also returns the dictionary with all of the cut dataframes (not scaled or logged)
Any values that create a -inf or inf log result will get replaced with NaN and interpolated before feeding into ss
"""
def exo_date_select_interpolate(dict, target_name, start_date, end_date, log=None, scaler='ss', freq='MS'):
    date_frame = sort_by_date(dict)
    names = list(date_frame['File Name'])
    
    cut_dfs = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    for n in range(len(names)):
        cut_dfs[names[n]] = pd.DataFrame(dict[names[n]])
        cut_dfs[names[n]] = cut_dfs[names[n]].reindex(date_range)
        cut_dfs[names[n]] = cut_dfs[names[n]].interpolate(method='linear', limit_direction='both')
    
    target = cut_dfs[target_name]
    cut_dfs.pop(target_name, None)
    exo_names = list(cut_dfs.keys())

    exo = pd.concat(cut_dfs.values(), axis=1)
    exo.index = date_range
    exo.columns = exo_names

    exo = exo.apply(pd.to_numeric)
    target = target.apply(pd.to_numeric)

    if log == 'log':
        exo = np.log(exo)
        target = np.log(target)
    elif log == 'log10':
        exo = np.log10(exo)
        target = np.log10(target)
    elif log == 'log2':
        exo = np.log2(exo)
        target = np.log2(target)
    else:
        pass   

    # Replace problematic values with a nan and then interpolate
    exo_clean = exo.copy()
    exo_clean[exo_clean == -math.inf] = np.nan
    exo_clean[exo_clean == math.inf] = np.nan
    exo_clean = exo_clean.interpolate(method='linear', limit_direction='forward')

    if scaler == 'ss':
        exo_s = ss_scale_a_df(exo_clean, freq=freq)
        target_s = ss_scale_a_df(target, freq=freq)
        
    elif scaler == 'mm':
        exo_s = mm_scale_a_df(exo_clean, freq=freq)
        target_s = mm_scale_a_df(target, freq=freq)
    else:
        exo_s = exo_clean
        target_s = target
    
    return target_s, exo_s, cut_dfs

"""
Function: full_exo_var_analysis
Combines all of the intermediate steps of auto-collecting and sorting exogenous variables when compared to a target.
Returns exo_features, a dataframe with each unique variable that was above the threshold, the best lagetime associated with it, 
and the correlation of that lag-variable pair. 
Also returns the selected_dictionary which is a dictionary containing the dfs for the features that were selected. (not scaled or logged)
Think of exo_features like a description of what is inside selected_dictionary and the best way to use each df inside
"""
def full_exo_var_analysis(cut_dfs, target_df, exo_df, start, end, ignore=None, thresh=0.5, show_num=1, freq='MS', lag_length=25):
    target_lag = lag_a_df(target_df, start_date=start, end_date=end, freq=freq, lag_end=lag_length)
    sorted_corr_target = threshold_correlation(target_lag, exo_df, ignore=ignore, thresh=thresh)
    exo_var_dict = variable_extraction(sorted_corr_target, show_num=show_num)
    
    selected_dictionary = {}
    selects = list(exo_var_dict.keys())
    for i in range(len(selects)):
        selected_dictionary[selects[i]] = cut_dfs[selects[i]]
    
    exo_features = pd.concat(exo_var_dict.values(), axis=0)

    return exo_features, selected_dictionary


"""
Pair of functions that make a df of only monthly frequency dfs from a dict
"""
def is_monthly_frequency(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    
    freq = pd.infer_freq(df.index)
    return freq == 'MS'

def filter_monthly_dataframes(dataframes_dict):
    monthly_dataframes = {key: df for key, df in dataframes_dict.items() if is_monthly_frequency(df)}
    return monthly_dataframes