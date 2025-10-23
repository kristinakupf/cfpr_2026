import numpy as np
import pandas as pd
import pathlib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
)
from sklearn.preprocessing import StandardScaler
from autogluon.timeseries import TimeSeriesPredictor

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


##################################################### EXPERIMENT FORECAST FUNCTIONS #############################################################################

################## Data loading and filtering ################################################

def load_data(target_categories, file_path):
              
    all_data = pd.read_csv(file_path, index_col=0)

    all_data.index = pd.to_datetime(all_data.index)
    all_data = all_data.asfreq(freq='MS')
    # all_data.interpolate(method='linear')
    if 'target_categories' == 'all':
        print('renaming')
        target_categories = [col for col in all_data.columns if col.startswith("food_cpi")]
    foodprice_df = all_data[target_categories]

    all_covariates = all_data.drop(columns=target_categories)

    return all_data, foodprice_df, target_categories, all_covariates

# Function to split dataframe into individual columns and store in a dictionary
def split_dataframe(df):

    df_dict = {}
    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        last_valid_index = df[column].last_valid_index()

        cleaned_col = df.loc[first_valid_index:last_valid_index, column]
        df_dict[column] = pd.DataFrame(cleaned_col, columns=[column])
    return df_dict

def filter_data_exp_1(all_data, start_year, end_date="2024-07-01"):
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(end_date)
    
    all_data_dict = split_dataframe(all_data)
    all_data_list = []
    
    for column in list(all_data_dict.keys()):
        df = pd.DataFrame(all_data_dict[column])
        df.index = pd.to_datetime(df.index)
        
        if df.index[0] <= pd.to_datetime("1986-01-01") and df.index[-1] >= end_date:
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
            all_data_list.append(filtered_df)
    
    all_data = pd.concat(all_data_list, axis=1)
    return all_data

def filter_data_exp_2(all_data, start_year, end_date="2024-07-01"):
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(end_date)
    
    all_data_dict = split_dataframe(all_data)
    all_data_list = []
    
    for column in list(all_data_dict.keys()):
        df = pd.DataFrame(all_data_dict[column])
        df.index = pd.to_datetime(df.index)
        
        if df.index[0] <= start_date and df.index[-1] >= end_date:
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
            all_data_list.append(filtered_df)
    
    all_data = pd.concat(all_data_list, axis=1)
    return all_data

# Function to filter dictionary for fully available data using date range
def filter_and_concat_data(year, all_data_dict):
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime("2024-07-01")

    all_data_list = []

    for column in all_data_dict.keys():
        df = pd.DataFrame(all_data_dict[column])
        df.index = pd.to_datetime(df.index)

        if df.index[0] <= start_date and df.index[-1] >= end_date:
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
            all_data_list.append(filtered_df)

    all_data_df = pd.concat(all_data_list, axis=1)
    num_columns = all_data_df.shape[1]

    return num_columns


# Function to create table with number of df available for a time period
def display_fully_available_data(all_data):

    years = []
    num_columns_list = []
    all_data_dict = split_dataframe(all_data)

    for year in range(1986, 2025):
        num_columns = filter_and_concat_data(year, all_data_dict)
        years.append(year)
        num_columns_list.append(num_columns)

    info_df = pd.DataFrame(
        {
            "Year-2024": years,
            "Number of Columns": num_columns_list,
        }
    )
    return info_df




################## AutoGluon Formatting Processor ################################################

# Class that creates dfs each of the model type
class AutoGluonProcessor:
    def __init__(self, all_data, target_categories):
        self.all_data = all_data
        self.all_covariates = self.all_data
        self.all_covariates_list = self.all_data.drop(columns=target_categories)
        self.all_data.index = pd.to_datetime(self.all_data.index)
        self.target_categories = target_categories

    def scale_df(self, df):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    def get_autogluon_local_df(self, study_category, cutoff_date):

        study_df = self.all_data[self.all_data.index <= cutoff_date][[study_category]]
        return (
            study_df.reset_index()
            .melt(id_vars="index", var_name="item_id", value_name="target")
            .rename({"index": "timestamp"}, axis=1)
        )

    def get_autogluon_local_with_covariates_df(self, study_category, cutoff_date):

        other_categories = [cat for cat in self.all_covariates_list]
        study_df_ag = self.get_autogluon_local_df(study_category, cutoff_date)

        covar_df_scaled = self.scale_df(
            self.all_data[self.all_data.index <= cutoff_date]
        )
        new_columns = {
            f"exogenous_{cat}": covar_df_scaled[cat][study_df_ag.timestamp].values
            for cat in other_categories
        }
        return pd.concat(
            [study_df_ag, pd.DataFrame(new_columns, index=study_df_ag.index)], axis=1
        )

    def get_autogluon_global_df(self, cutoff_date):

        food_cpi = pd.concat(
            [
                self.get_autogluon_local_df(cat, cutoff_date)
                for cat in self.target_categories
            ],
            axis=0,
        )
        covar_df = self.scale_df(
            self.all_data[self.all_data.index <= cutoff_date].drop(
                columns=self.target_categories
            )
        )
        covar_df_ag = (
            covar_df.reset_index()
            .melt(id_vars="index", var_name="item_id", value_name="target")
            .rename({"index": "timestamp"}, axis=1)
        )
        return pd.concat([food_cpi, covar_df_ag], axis=0)

    def get_autogluon_global_with_covariates_df(self, cutoff_date):

        return pd.concat(
            [
                self.get_autogluon_local_with_covariates_df(cat, cutoff_date)
                for cat in self.target_categories
            ],
            axis=0,
        )

    def get_autogluon_local_df_filt(self, study_category, cutoff_date, filt_regressors):

        
        self.all_data_filt = self.all_data[filt_regressors+self.target_categories]
        study_df = self.all_data_filt[self.all_data_filt.index <= cutoff_date][[study_category]]

        return (
            study_df.reset_index()
            .melt(id_vars="index", var_name="item_id", value_name="target")
            .rename({"index": "timestamp"}, axis=1)
        )

    def get_autogluon_local_with_covariates_df_filt(self, study_category, cutoff_date, filt_regressors):

        self.all_data_filt = self.all_data[filt_regressors+self.target_categories]
        all_covariates_list_filt = self.all_covariates_list[filt_regressors]       
        other_categories = [cat for cat in all_covariates_list_filt]
        study_df_ag = self.get_autogluon_local_df_filt(study_category, cutoff_date, filt_regressors)

        covar_df_scaled = self.scale_df(
            self.all_data_filt[self.all_data_filt.index <= cutoff_date]
        )
        new_columns = {
            f"exogenous_{cat}": covar_df_scaled[cat][study_df_ag.timestamp].values
            for cat in other_categories
        }
        return pd.concat(
            [study_df_ag, pd.DataFrame(new_columns, index=study_df_ag.index)], axis=1
        )    
    
    def get_autogluon_global_df_filt(self, cutoff_date, filt_regressors):

        food_cpi = pd.concat(
            [
                self.get_autogluon_local_df_filt(cat, cutoff_date, filt_regressors)
                for cat in self.target_categories
            ],
            axis=0,
        )
        covar_df = self.scale_df(
            self.all_data_filt[self.all_data_filt.index <= cutoff_date].drop(
                columns=self.target_categories
            )
        )
        covar_df_ag = (
            covar_df.reset_index()
            .melt(id_vars="index", var_name="item_id", value_name="target")
            .rename({"index": "timestamp"}, axis=1)
        )
        return pd.concat([food_cpi, covar_df_ag], axis=0)

    def get_autogluon_global_with_covariates_df_filt(self, cutoff_date, filt_regressors):

        return pd.concat(
            [
                self.get_autogluon_local_with_covariates_df_filt(cat, cutoff_date, filt_regressors)
                for cat in self.target_categories
            ],
            axis=0,
        )



########################## EXPERIMENT LOOP: Predictor ###################################################################

# Function to get the model list based on the base type
def get_model_list(EXP_BASE):

    if "ag_local" in EXP_BASE:
        return [
            "NaiveModel",
            "SeasonalNaiveModel",
            "AutoARIMAModel",
            "AutoETSModel",
            "DeepARModel",
            "DLinearModel",
            "PatchTSTModel",
            "SimpleFeedForwardModel",
            "TemporalFusionTransformerModel",
#             "DirectTabularModel",
#             "RecursiveTabularModel",
            "ChronosModel",
        ]
    elif "ag_global_all" in EXP_BASE :
        return [
            "DeepARModel",
            "DLinearModel",
            "PatchTSTModel",
            "SimpleFeedForwardModel",
            "TemporalFusionTransformerModel",
#             "DirectTabularModel",
#             "RecursiveTabularModel",
            "ChronosModel",
        ]
    elif ("ag_global_cpi" in EXP_BASE) & ("covariates" not in EXP_BASE):
        return [
            "DeepARModel",
            "DLinearModel",
            "PatchTSTModel",
            "SimpleFeedForwardModel",
            "TemporalFusionTransformerModel",
#             "DirectTabularModel",
#             "RecursiveTabularModel",
            "ChronosModel",
        ]
    elif "ag_global_cpi_with_covariates" in EXP_BASE:
        return [
            "TemporalFusionTransformerModel"
        ]
    else:
        return []


# Create the directories for the results of each experiments
def create_output_directories(
        exp_test_description, start_year, EXPERIMENT_NAME, cutoff_date, random_seed
):
    if random_seed == 123:

        forecast_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/{cutoff_date}/forecasts"
        plot_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/{cutoff_date}/plots"
        training_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/{cutoff_date}/training_results"
        model_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/{cutoff_date}/model_files/"

        # Create the directories if they don't exist
        pathlib.Path(forecast_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(training_output_dir).mkdir(parents=True, exist_ok=True)

    else:
        forecast_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/random_seed_{random_seed}/{cutoff_date}/forecasts"
        plot_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/random_seed_{random_seed}/{cutoff_date}/plots"
        training_output_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/random_seed_{random_seed}/{cutoff_date}/training_results"
        model_dir = f"./output/{exp_test_description}/{EXPERIMENT_NAME}/random_seed_{random_seed}/{cutoff_date}/model_files/"

        # Create the directories if they don't exist
        pathlib.Path(forecast_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(training_output_dir).mkdir(parents=True, exist_ok=True)

    return forecast_output_dir, plot_output_dir, training_output_dir, model_dir



def fit_predictors_local(model_type, model_dir, train_data, category, random_seed):

    time_limit = 1800
    num_val_windows = 3
    
    predictor = TimeSeriesPredictor(
        prediction_length=18,
        log_to_file = False,
        path=f"{model_dir}/{category}/",
        target="target",
        eval_metric="MAPE",
        quantile_levels=[0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99],
    )
   


    print(model_type)
    if  "Chronos" in model_type:
         model_params = {"model_path": "autogluon/chronos-t5-large"}  
         predictor.fit(
                    train_data,
                    hyperparameters={model_type: model_params},
#                     time_limit=1800,
                    time_limit=time_limit,
                    excluded_model_types=["DirectTabular", "RecursiveTabular"],
                    num_val_windows=num_val_windows,
                    random_seed=random_seed
                )    
        
    else:
        model_params = {}
        predictor.fit(
            train_data,
            hyperparameters={model_type: model_params},
    #         excluded_model_types=["DirectTabular"],
            time_limit=1800,
            random_seed=random_seed
        )


    return predictor

def fit_predictors_global(model_type, model_dir, train_data, random_seed):

    max_num_items = 100000
    time_limit = 1800
    num_val_windows = 3
    predictor = TimeSeriesPredictor(
        prediction_length=18,
        log_to_file = False,
        path=model_dir,
        target="target",
        eval_metric="MAPE",
        quantile_levels=[0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99],
    )
    if model_type == "DirectTabularModel":
        model_params = {"max_num_items": max_num_items}  
        predictor.fit(
            train_data,
            hyperparameters={model_type: model_params},
#             time_limit=1800,
            time_limit=time_limit,
            excluded_model_types=["DirectTabular", "RecursiveTabular"],
            num_val_windows=num_val_windows,
            random_seed=random_seed
        )

    print(model_type)
    if  "Chronos" in model_type:
         model_params = {"model_path": "autogluon/chronos-t5-large"}  
         predictor.fit(
                    train_data,
                    hyperparameters={model_type: model_params},
#                     time_limit=1800,
                    time_limit=time_limit,
                    excluded_model_types=["DirectTabular", "RecursiveTabular"],
                    num_val_windows=num_val_windows,
                    random_seed=random_seed
                )    
    else:
        model_params = {}
        predictor.fit(
            train_data,
            hyperparameters={model_type: model_params},
#             time_limit=1800,
            time_limit=time_limit,
            excluded_model_types=["DirectTabular", "RecursiveTabular"],
            num_val_windows=num_val_windows,
            random_seed=random_seed
        )
    return predictor




############################## PLotting Functions ################################################

# Plot the individual forecast for each of the categories and for each of the cut-off dates
def plot_quantile_forecast(
    category,
    context_df,
    forecast_df,
    actual_df,
    cutoff_date,
    save_path=None,
    show_plots=True,
    model_name="",
):

    fig, ax = plt.subplots(figsize=(10, 6))

    context_df = pd.concat((context_df, actual_df))
    ax.plot(context_df.index, context_df.values, color="black", label="Historical CPI")

    ax.fill_between(
        forecast_df.index,
        forecast_df[f"q_0.05"],
        forecast_df[f"q_0.95"],
        facecolor="purple",
        alpha=0.5,
        label="95% Confidence",
    )

    ax.plot(
        forecast_df.index,
        forecast_df[f"q_0.5"],
        color="purple",
        label="Median Forecast",
    )

    ax.set_title(f"{category}\nRetrospective Forecast - {cutoff_date}\n{model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI (% 2002 Prices)")
    ax.axvline(
        pd.to_datetime(cutoff_date),
        label="Cutoff Date",
        color="black",
        ls="--",
        ms=1,
        alpha=0.5,
    )
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    if show_plots:
        plt.show()

    if save_path:
        fig.savefig(
            save_path, dpi=300 if save_path.endswith("png") else None
        ) 


def process_and_plot_forecasts(predictor, train_data, category, training_output_dir, forecast_output_dir,plot_output_dir, foodprice_df,cutoff_date):

    with open(f"{training_output_dir}/{category}.txt", 'w') as f:
        f.write(str(predictor.fit_summary()))
    forecast_df = predictor.predict(train_data).loc[category]

    
    forecast_df = predictor.predict(train_data).loc[category]
    forecast_df = forecast_df.rename({**{"mean": "q_0.5"}, **{col: f"q_{col}" for col in forecast_df.columns if col != "mean"}},
        axis=1
    )

    context_df = foodprice_df[category].loc[(foodprice_df.index >= pd.to_datetime(cutoff_date) - pd.DateOffset(months=120)) & (foodprice_df.index <= cutoff_date)]
    actual_df = foodprice_df[category].loc[(foodprice_df.index > cutoff_date) & (foodprice_df.index <= forecast_df.index.max())]

    forecast_df.to_csv(f"{forecast_output_dir}/{category}.csv")

    plot_quantile_forecast(
        category=category,
        context_df=context_df,
        forecast_df=forecast_df,
        actual_df=actual_df,
        save_path=f"{plot_output_dir}/{category}.svg",
        cutoff_date=cutoff_date,
        show_plots=True,
        model_name=predictor.model_best
    )





##################################################### EXPERIMENT ANALYSIS #############################################################################



################################### Evaluation Metrics ################################################
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pinball_loss(y_true, y_pred_quantiles, quantiles):
    losses = {}
    for q in quantiles:
        col = f"q_{q}"
        y_pred = y_pred_quantiles[col].values
        pb_loss = mean_pinball_loss(y_true, y_pred, alpha=q)
        losses[col] = pb_loss
    return pd.Series(losses)


def average_pinball_loss(y_true, y_pred_quantiles, quantiles):
    return pinball_loss(y_true, y_pred_quantiles, quantiles).mean()



################################### Plotting Functions #################################################################

def plot_forecasts_analysis(category, context_df, forecast_dfs_by_method, actual_df, cutoff_dates, save_path=None, show_plots=True):

   
    fig = plt.figure(figsize=(12,6))
    plot_ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # left, bottom, width, height (range 0 to 1)
    legend_ax = fig.add_axes([0.77, 0.1, 0.2, 0.8])
    legend_ax.axis('off')  

    combined_df = pd.concat((context_df, actual_df))
    plot_ax.plot(combined_df.index, combined_df.values, color='black', label='Historical')

    colors = ['purple', 'orange', 'red']  

    for method_index, (method, forecast_dfs) in enumerate(forecast_dfs_by_method.items()):
        color = colors[method_index % len(colors)]

        for index, forecast_df in enumerate(forecast_dfs):
            # Confidence range between 0.05 and 0.95 quantiles (assuming these columns exist)
            plot_ax.fill_between(
                forecast_df.index,
                forecast_df["q_0.05"],
                forecast_df["q_0.95"],
                facecolor=color,
                alpha=0.2,
                label=f'95% Confidence ({method})' if index == 0 else None  # label only the first instance
            )

            plot_ax.plot(
                forecast_df.index,
                forecast_df["q_0.5"],
                color=color,
                # label=f'Median Forecast ({method})' if index == 0 else None  # label only the first instance
            )

    plot_ax.set_title(f'{category} Price Forecast Comparison')
    plot_ax.set_xlabel('Date')
    plot_ax.set_ylabel('Price')
    for cutoff_date in cutoff_dates:
        plot_ax.axvline(pd.to_datetime(cutoff_date), color='black', ls='--', lw=1, alpha=0.5)

    handles, labels = plot_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center left')

    plot_ax.xaxis.set_major_locator(mdates.YearLocator())
    plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plot_ax.grid(axis='y')


    if show_plots:
        plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300 if save_path.endswith("png") else None)  # High res for png

def plot_forecasts_analysis_vs(category, context_df, forecast_dfs_by_method, actual_df, cutoff_dates, param, save_path=None, show_plots=True):

   
    fig = plt.figure(figsize=(12,6))
    plot_ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # left, bottom, width, height (range 0 to 1)
    legend_ax = fig.add_axes([0.77, 0.1, 0.2, 0.8])
    legend_ax.axis('off')  

    combined_df = pd.concat((context_df, actual_df))
    plot_ax.plot(combined_df.index, combined_df.values, color='black', label='Historical')

    colors = ['purple', 'orange', 'red']  

    for method_index, (method, forecast_dfs) in enumerate(forecast_dfs_by_method.items()):
        color = colors[method_index % len(colors)]

        for index, forecast_df in enumerate(forecast_dfs):
            # Confidence range between 0.05 and 0.95 quantiles (assuming these columns exist)
            plot_ax.fill_between(
                forecast_df.index,
                forecast_df["q_0.05"],
                forecast_df["q_0.95"],
                facecolor=color,
                alpha=0.2,
                label=f'95% Confidence ({method})' if index == 0 else None  # label only the first instance
            )

            plot_ax.plot(
                forecast_df.index,
                forecast_df["q_0.5"],
                color=color,
                # label=f'Median Forecast ({method})' if index == 0 else None  # label only the first instance
            )

    plot_ax.set_title(f'{category} Price Forecast Comparison')
    plot_ax.set_xlabel('Date')
    plot_ax.set_ylabel('Price')
    for cutoff_date in cutoff_dates:
        plot_ax.axvline(pd.to_datetime(cutoff_date), color='black', ls='--', lw=1, alpha=0.5)

    handles, labels = plot_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center left')

    plot_ax.xaxis.set_major_locator(mdates.YearLocator())
    plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plot_ax.grid(axis='y')


    if show_plots:
        plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300 if save_path.endswith("png") else None)  # High res for png

def plot_quantile_forecast(category, context_df, forecast_df, actual_df, cutoff_date, save_path, show_plots, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    context_df = pd.concat((context_df, actual_df))
    ax.plot(context_df.index.to_numpy(), context_df.to_numpy(), color="black", label="Historical CPI")
    
    ax.fill_between(
        forecast_df.index.to_numpy(),
        forecast_df["q_0.05"].to_numpy(),
        forecast_df["q_0.95"].to_numpy(),
        color="blue",
        alpha=0.3,
        label="90% Confidence",
    )
    
    ax.plot(
        forecast_df.index.to_numpy(),
        forecast_df[f"q_0.5"].to_numpy(),
        color="purple",
        label="Median Forecast",
    )
    
    check_path = save_path.split('.food_cpi')[0]
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    
    ax.set_title(f"{category} Forecast - {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
#     plt.savefig(save_path)
    
    if show_plots:
        plt.show()

