# Repository: cfpr_2025

The purpose of this repository is to house Zohrah and Mya's contributions to Canada's Food Price Report as Undergraduate Research Assistants with the University of Guelph in Summer 2024

## Legend
|Folder|Folder Description|
|-|-|
| [cfpr_data_processing](cfpr_data_processing) | Holds notebook and functions for data processing |
| [cfpr_experiments](cfpr_experiments) | Holds outputs of CFPR forecasting experiment notebooks |
| [corr_variable_selection](corr_variable_selection) | Holds notebook and functions for correlation-based variable selection and provincial[^1] variable selection |
| [data](data) | Holds subfolders archived data[^2], processed_data, provincial data[^1], and raw data |
| [EDA](EDA) | Holds exploratory data analysis notebooks[^3] |

[^1]: May be removed depending on the course of the project
[^2]: To be removed upon sharing the repo 
[^3]: Not fully clean nor recently updated; can be removed if seen unhelpful

## General Workflow
* DATA
  * This repository holds pre-selected and pre-processed time series data for Canadian CPI food prices and a swath of exogenous regressor data in the [data folder](data). If new data is to be added through csv files, they can be run through the [processing notebook](cfpr_data_processing/data_processing.ipynb).
  * For further data selection and analysis, the [correlation-based variable selection notebook](corr_variable_selection/corr_variable_selection.ipynb) can be used.
* FORECASTING
  * [Window timing experiment](cfpr_experiments/window_timing_exp)
    * notebooks separated by experiments and year
  * [Variable selection experiment](cfpr_experiments/variable_selection_exp)
   

## Appendix
### Acronyms Used for Database Identification in [data](data)

| Acronym | Database |
| ------- | -------- |
| BC |  Bank of Canada |
| CDEC |  California Data Exchange Center |
| EPU | Economic Policy Uncertainty |
| FRED | Federal Reserve Economic Data |
| GT | Google Trends |
| IMF | International Monetary Fund |
| NCEI |  National Centers for Environmental Information |
| NOAA |  National Oceanic and Atmospheric Administration |
| NYFED |  Federal Reserve Bank of New York |
| STATSCAN | Stats Canada |
| WB |  World Bank |

### Changes to make to the repo before sharing it!
* Adding a license
* Clearing out old / irrelevant files
* Making sure all the paths are working correctly with merge

