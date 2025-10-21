# 🧾 CFPR 2025 Forecast Evaluation Pipeline

## 📘 Overview

This repository contains code and data to **evaluate and compare time-series forecasting models** used in the 2025 *Canada Food Price Report (CFPR)*.  
It generates a comprehensive Markdown report summarizing:

- **Monthly accuracy** (Mean Absolute Percentage Error, *MAPE*)  
- **Year-over-Year (YoY) forecast accuracy**  
- **Cross-analysis** of model agreement across metrics  
- **Visual percent-error trends** for each food category  

The goal is to identify the **most reliable forecasting models** and track their performance consistency across categories and time periods.

---

## 📂 Repository Structure

cfpr_2026/
├── data/
│ ├── raw_data_updated/STATSCAN_food_cpi.csv
│ └── processed_data_updated/STATSCAN_food_cpi_processed.csv
│
├── generate_forecasts/output/Forecasts/
│ └── <model_name>/random_seed_42/2024-07-01/forecasts/food_cpi: <category>.csv
│
├── cfpr_evaluation_dicts/
│ ├── category_results_mape.pkl
│ ├── category_residual_results.pkl
│ ├── combined_summaries.pkl
│ ├── overall_summary_mape.csv
│ ├── overall_summary_yoy.csv
│ ├── overall_summary_cross.csv
│ └── model_coverage_cross.csv
│
├── cfpr_figures_percent_error_dynamic/
│ ├── Food_percent_error_dynamic.png
│ ├── Meat_percent_error_dynamic.png
│ └── ...
│
├── reports/
│ ├── cfpr_forecast_eval_report_2025-10-21.md
│ └── README.md ← (this file)
│
└── scripts/
├── monthly_eval.ipynb
├── yoy_eval.ipynb
├── cross_analysis.ipynb
└── generate_cfpr_eval_report.py
    

---

## 🧮 Step-by-Step: How to Generate the Evaluation Report

### 1️⃣ Monthly Evaluation (MAPE)
**Notebook:** `monthly_eval.ipynb`

- Loads ground-truth CPI data (`STATSCAN_food_cpi_processed.csv`)  
- Loads forecasts for all models under `generate_forecasts/output/Forecasts/`  
- Computes monthly **MAPE** for each food category  
- Produces:
  - `category_results_mape.pkl`  
  - `overall_summary_mape.csv`  
  - Percent-error plots saved in `cfpr_figures_percent_error_dynamic/`

---

### 2️⃣ Year-over-Year Evaluation
**Notebook:** `yoy_eval.ipynb`

- Calculates actual YoY change (2025 vs. 2024)  
- Calculates forecasted YoY change per model  
- Computes residuals (forecast − actual YoY %)  
- Produces:
  - `category_residual_results.pkl`  
  - `overall_summary_yoy.csv`

---

### 3️⃣ Cross-Metric Analysis
**Notebook:** `cross_analysis.ipynb`

- Loads both MAPE and YoY dictionaries  
- Combines them to find models performing consistently well on both metrics  
- Produces:
  - `combined_summaries.pkl`  
  - `overall_summary_cross.csv`  
  - `model_coverage_cross.csv`

---

### 4️⃣ Report Generation
**Script:** `scripts/generate_cfpr_eval_report.py`

Generates a GitHub-renderable Markdown report summarizing all results.

Run:
```bash
python scripts/generate_cfpr_eval_report.py
    
Outputs:

/reports/cfpr_forecast_eval_report_<date>.md

Embeds plots from cfpr_figures_percent_error_dynamic/

Includes top models, tables, and recommendations per category and overall.
