#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils_experiment_1800 import (
    load_data,
    filter_data_exp_1,
    AutoGluonProcessor,
    fit_predictors_local,
    fit_predictors_global,
    get_model_list,
    create_output_directories,
    process_and_plot_forecasts,
)


# In[2]:


import torch
print("Is cuda available? {}".format(torch.cuda.is_available()))
torch.cuda.get_device_name(torch.cuda.current_device())


# ## Experiment 1 Details
# 
# Experiment 1 serves as a baseline for comparing the effect of different window lengths for training. The data will be filtered to include only time series that are available in the time period. To investiage the effect of the time series by themselves, the number of covariates is kept constant across all time series and corresponds to the time series fully available from 1986. The specific time periods were chosen with the condition that between each interval, 5 or more new time series were introduced.
# 
# | Time Period - 2024     | Maximum number of covariates | Number of covariates used |
# |--------------|:-----:| :-----:| 
# | 1986 |   94  |   94  |  
# | 1997 |   111  |  94  |  
# | 2004 |   133  |   94  |  
# | 2007 |   146  |  94  |  
# | 2010 |   154  |  94  |  
# | 2018 |   159  |  94  |  
# 
# 
# The cut-off dates are set at 4 yearly intervals between 2020 and 2024. Since the minimum training length for a yearly interval is 2 years, 2020 is the earliest cut-off date we can use.
# 

# In[3]:


exp_test_description = "Full_Report"
report_sim_dates = open("/h/kupfersk/cfpr_2025/generate_forecasts/experiment_cutoff_dates.txt", 'r').read().split()
report_sim_dates=[report_sim_dates[0]]

target_categories = [
     'Fruit, fruit preparations and nuts',
     'Meat',
     'Other food products and non-alcoholic beverages',
     'Vegetables and vegetable preparations',    
     'Bakery and cereal products (excluding baby food)',
     'Dairy products and eggs',
     'Fish, seafood and other marine products',
     'Food purchased from restaurants',
     'Food',
]


target_categories = [f"food_cpi: {col}" for col in target_categories]

start_year = 1986
random_seed = 42

print("Target Categories:\n" + '\n'.join(target_categories))
print("Report Simulation Dates:\n" + '\n'.join(report_sim_dates))


# ## Load data
# 
# `all_data`: dataframe with all the food cpi variables and all the covariates from 1986 to 2024.\
# `foodprice_df`: dataframe with only the food cpi variables from 1986 to 2024.\
# `target_categories`: list of the names of the food cpi variables.\
# `all_covariates`: dataframe with all the covariates from 1986 to 2024.

# In[52]:


file_path = "/h/kupfersk/cfpr_2025/data/processed_data/all_select.csv"
all_data, foodprice_df, target_categories, all_covariates = load_data(target_categories, file_path)


# In[53]:


# all_data = filter_data_exp_1(all_data, start_year)
processor = AutoGluonProcessor(all_data, target_categories)


# 
# # Experiment List
# 
# | AutoGluon Model \ Experiment      | local | global | global + covariates |
# |--------------|:-----:| :-----: | :-----: |
# | NaiveModel |   x  |   |  | 
# | SeasonalNaiveModel|   x    |  |  | 
# | AutoARIMAModel|   x   |  |  | 
# | AutoETSModel|  x  |    |  | 
# | DeepARModel|  x   | x |  x | 
# | DLinearModel| x   | x |  x| 
# | PatchTSTModel|  x   | x | x |   
# |SimpleFeedForwardModel|  x   | x | x |   
# | TemporalFusionTransformerModel| x  | x | x |   
# | DirectTabularModel| x   | x | x | 
# |RecursiveTabularModel|  x   | x | x |  
# |ChronosModel|  x   | x | x | 
# 
# x - model type supported
# 

# # Main experiment loop - global models
# 
# 5 main loops:
# - The first loop goes through the list of years at which we want the time window to start, filters out the covariates that are not fully available from 1986.
# - The second loop goes though the two experiment base (global and global+covariates)
# - The third loop goes though 8 Autogluon models, excluding all statistical models (Naive, SeasonalNaive, AutoArima, AutoETS).
# - The fourth loop goes through each of the yearly cut-off dates to trim the training data.
# - The fifth loop goes through each of the 9 food categories, gets the training data in the AutoGluon format depending on the experiment base and plot each prediction period againt the actual values.

# In[54]:


import json

# Load the dictionary from the JSON file
with open('experiment_dict_filt.json', 'r') as json_file:
    experiment_dict = json.load(json_file)
    
experiment_dict


# In[ ]:


models_to_test = [
    'ChronosModel',
    'DeepARModel',
]

EXP_BASE_LIST_GLOBAL = ["ag_global_all", "ag_global_cpi_with_covariates"]


for EXP_BASE in EXP_BASE_LIST_GLOBAL:
    EXP_MODEL_LIST = get_model_list(EXP_BASE)
    print(EXP_MODEL_LIST)

    for model_type in EXP_MODEL_LIST:
        
        if model_type not in models_to_test:
            continue
            print(model_type)
                

        EXPERIMENT_NAME = f"{EXP_BASE}_{model_type}"

        for category in (target_categories):
            print(model_type)

            
            #For this specific category, collect the experiments that need to be run by filtering the data 
            FILT_EXPS = experiment_dict[category.split("food_cpi: ")[1]]
            
            for cutoff_date in report_sim_dates:
                
                for FILT_NUM in FILT_EXPS:
                    MOD_FILT_NUM = FILT_NUM.replace(category.split("food_cpi: ")[1], "").split('.txt')[0]
                    MOD_EXP_NAME = f"{EXPERIMENT_NAME}_{MOD_FILT_NUM}"
                    print(FILT_NUM)
                    
                    with open(f"/h/kupfersk/cfpr_2025/cfpr_variable_selection/Full_Report_Experiments/{FILT_NUM}", 'r') as file: 
                        # Use .strip() to remove any trailing newline or spaces
                        item_list = [line.strip() for line in file.readlines()]
                        filt_regressors = item_list
                        
                    forecast_output_dir, plot_output_dir, training_output_dir, model_dir = (
                        create_output_directories(
                            exp_test_description,
                            start_year,
                            MOD_EXP_NAME,
                            cutoff_date,
                            random_seed
                        )
                    )

                    if "_covariates" in EXP_BASE:
                        train_data = processor.get_autogluon_global_with_covariates_df_filt(
                            cutoff_date, filt_regressors
                        )
                    else:
                        train_data = processor.get_autogluon_global_df_filt(cutoff_date, filt_regressors)

                    predictor = fit_predictors_global(model_type, model_dir, train_data, random_seed)


                    process_and_plot_forecasts(
                        predictor,
                        train_data,
                        category,
                        training_output_dir,
                        forecast_output_dir,
                        plot_output_dir,
                        foodprice_df,
                        cutoff_date
                    )


# In[ ]:




