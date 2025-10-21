import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from dateutil import parser
import re

raw_path = "../data/raw_data/"
processed_path = "../data/processed_data/"


# Convert date format to "%Y-%m-%d"
def convert_date(df, column_name):
    def convert(input_date):
        input_date = str(input_date)  # Convert input to string
        if not isinstance(input_date, str):
            return "Invalid date format"
        try:
            # Handle "1990" and "1990.0" formats
            year_match = re.match(r"^(\d{4})(\.0)?$", input_date)
            if year_match:
                year = year_match.group(1)
                input_date = f"{year}-01-01"

            # Handle "Q4 1990" format
            quarter_match = re.match(r"Q([1-4]) (\d{4})", input_date)
            if quarter_match:
                quarter, year = quarter_match.groups()
                month = (int(quarter) - 1) * 3 + 1  # Convert quarter to starting month
                input_date = f"{year}-{month:02d}-01"

            # Handle "Jun-90" format
            month_year_match = re.match(r"([A-Za-z]{3})-(\d{2})", input_date)
            if month_year_match:
                month, year = month_year_match.groups()
                if int(year) < 50:  # Handle years in 2000s
                    year = f"20{year}"
                else:  # Handle years in 1900s
                    year = f"19{year}"
                # input_date = f"01 {month} {year}"
                input_date = f"{year}-{month}-01"

            # Handle "January 1990" format
            full_month_year_match = re.match(r"([A-Za-z]+) (\d{4})", input_date)
            if full_month_year_match:
                month, year = full_month_year_match.groups()
                input_date = f"{year}-{month}-01"

            # Handle "1990M01" format
            full_year_m_match = re.match(r"(\d{4})M(\d{2})", input_date)
            if full_year_m_match:
                year, month = full_year_m_match.groups()
                input_date = f"{year}-{month}-01"

            # Handle "1990-01" format
            year_month_match = re.match(r"(\d{4})-(\d{2})", input_date)
            if year_month_match:
                year, month = year_month_match.groups()
                input_date = f"{year}-{month}-01"

            # Handle "31-Jan-1990" format
            day_month_year_match = re.match(
                r"(\d{2})-([A-Za-z]{3})-(\d{4})", input_date
            )
            if day_month_year_match:
                day, month, year = day_month_year_match.groups()
                input_date = f"{year}-{month}-01"

            date = parser.parse(input_date)
            return date.strftime("%Y-%m-%d")
        except ValueError:
            return "Invalid date format"

    df[column_name] = df[column_name].apply(convert)
    return df


# Changing frequency to monthly
def frequency_resampling(df):

    freq = pd.infer_freq(df.index)
    if freq != "MS":
        df = df.infer_objects()
        df = df.resample("MS").asfreq()
    return df


def plot_dataframe(df):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.legend()
    plt.xlabel("Index")
    plt.gcf().autofmt_xdate()  # Automatically rotate date labels
    plt.ylabel("Values")
    plt.title("Plot of DataFrame v/s Date")
    plt.show()


# drops all the columns with less than five values
def drop_columns_with_few_non_nulls(df, threshold=5):
    non_null_counts = df.notna().sum()
    columns_to_drop = non_null_counts[non_null_counts <= threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned


# Filter rows where any column except the first contains letters or symbols
def filter_rows_containing_letters_or_symbols(df):
    def contains_letters_or_symbols(s):
        return bool(re.search(r'[a-zA-Z!@#$%^&*()_+{}|:"<>?]', s))

    columns_to_check = df.columns[1:]
    df_filtered = df[
        ~df[columns_to_check].apply(
            lambda row: row.astype(str).apply(contains_letters_or_symbols).any(), axis=1
        )
    ]
    return df_filtered


# swap the order of 2 columns
def swap_columns(df, col1, col2):
    cols = df.columns.tolist()
    idx1, idx2 = cols.index(col1), cols.index(col2)
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
    df = df[cols]
    return df


# to remove subscripts in headers
def remove_subscripts(text):
    result = re.sub(r"\s\d+", " ", text)
    result = result.strip()
    return result


# to remove [] in headers
def remove_reg(text):
    return re.sub(r"\[.*?\]", "", text)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", "")
        df[col] = df[col].replace({"nan": pd.NA, "..": pd.NA})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = [remove_reg(col) for col in df.columns]
    df.columns = [remove_subscripts(col) for col in df.columns]

    return df


def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i, index in enumerate(dup_indices):
            cols[index] = f"{dup}_{i}" if i != 0 else dup
    df.columns = cols
    return df


def remove_capitals(text):
    new_text = re.sub(r"[A-Z]", "", text)
    return new_text


# saves files after processing
def save_processed_file(df, file_name, processed_path):
    newfilename = file_name.replace(".csv", "_processed.csv")
    new_file_path = os.path.join(processed_path, newfilename)
    df.to_csv(new_file_path, index=True)


# combines all the logistic df cleaning that is consistent across datasets
def finalize_df(df):
    df = convert_date(df, df.columns[0])
    df = df.rename(columns={df.columns[0]: "index"})
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%Y-%m-%d")
    df.set_index(df.columns[0], inplace=True)
    df = frequency_resampling(df)
    df = clean_numeric_columns(df)
    return df


# for splitting the all_df into a dictionary
def split_dataframe(df):
    df_dict = {}

    for column in df.columns:
        first_valid_index = df[column].first_valid_index()
        last_valid_index = df[column].last_valid_index()

        cleaned_col = df.loc[first_valid_index:last_valid_index, column]

        df_dict[column] = pd.DataFrame(cleaned_col, columns=[column])

    return df_dict


# ---------------------------Database Loading---------------------------#


# load data from statscan
def load_statscan(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.DataFrame(
        pd.read_csv(
            file_path,
            encoding="unicode_escape",
            skipfooter=13,
            skiprows=7,
            on_bad_lines="skip",
            engine="python",
            thousands=",",
            na_values="F",
        )
    )

    words_to_drop = [
        "Geography",
        "Sector",
        "Variable",
        "Reference period",
        "Prices",
        "Seasonal adjustment",
        "Type of employees",
        "Overtime",
        "Sales",
        "Adjustments",
        "Trade",
        "Dollars",
        "Current dollars",
        "Index",
        "Index, 2018=100",
        "Index, 2021=100",
        "Number",
        "Tonnes",
        "Metric tonnes",
        "2002=100",
        "Kilolitres",
    ]

    def row_contains_exact_strings(row, strings):
        for value in row:
            if isinstance(value, str) and value in strings:
                return True
        return False

    df = df[
        ~df.apply(lambda row: row_contains_exact_strings(row, words_to_drop), axis=1)
    ]
    headers = df.iloc[0].tolist()
    df = df.drop(df.index[0])
    df.columns = headers

    df = drop_columns_with_few_non_nulls(df, threshold=5)
    df = df.dropna(axis=0)
    df = filter_rows_containing_letters_or_symbols(df)

    df = finalize_df(df)
    save_processed_file(df, file_name, processed_path)

    return df


# load data from imf
def load_imf(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(
        file_path,
        encoding="unicode_escape",
        on_bad_lines="skip",
        engine="python",
        thousands=",",
        na_values="F",
    )
    df = df.transpose().dropna(axis=1).reset_index(drop=False)
    new_title = df.iloc[0, 0]
    df.columns = [df.columns[1], new_title]
    df = df.drop(df.index[0]).replace("no data", pd.NA).dropna()

    df = finalize_df(df)
    save_processed_file(df, file_name, processed_path)

    return df


# load data from fred
def load_fred(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, encoding="unicode_escape").dropna()

    df = finalize_df(df)

    save_processed_file(df, file_name, processed_path)

    return df


# load data from world bank
def load_world_bank(file_path, processed_path):
    file_name = os.path.basename(file_path)

    if file_name == "WB_world_crude_oil.csv":
        df = pd.read_csv(file_path, skiprows=4, header=0, na_values="…")
        df.drop(columns=df.columns[2:], inplace=True)
        df.drop(df.index[0:2], inplace=True)

    elif file_name == "WB_commodity_price_index.csv":
        df = pd.read_csv(
            file_path,
            header=9,
            na_values="…",
            skipfooter=579,
            engine="python",
            on_bad_lines="skip",
        )

    else:
        df = pd.read_csv(file_path, encoding="unicode_escape")
        new_title = df.iloc[4, 2]
        df = df.iloc[[3, 39]]
        df = df.dropna(axis=1)
        df = df.iloc[:, 4:]
        df = df.transpose()
        df.columns = [df.columns[0], new_title]

    df = finalize_df(df)
    df = drop_columns_with_few_non_nulls(df, threshold=5)
    save_processed_file(df, file_name, processed_path)

    return df


# load data from Google Trends
def load_google_trend(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, encoding="unicode_escape", header=2).dropna()

    df.replace("<1", 0.0001, inplace=True)

    df = finalize_df(df)
    save_processed_file(df, file_name, processed_path)

    return df


# load data from Economic policy uncertainty
def load_Economic_policy_uncertainty(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, encoding="utf-8-sig").dropna()
    df = df.drop(df.index[-1])

    df["Month"] = df["Month"].astype(int)
    df["Date"] = df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    df.drop(["Year", "Month"], axis="columns", inplace=True)

    df = swap_columns(df, df.columns[0], df.columns[1])

    df = finalize_df(df)
    save_processed_file(df, file_name, processed_path)

    return df


# load data from NOAA
def load_noaa(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        header=None,
        skipfooter=4,
        na_values="-999.00",
    )
    df = df.melt(id_vars=[0], var_name="Month", value_name="ENSO")
    df["index"] = pd.to_datetime(
        df[0].astype(str) + "-" + df["Month"].astype(str), format="%Y-%m"
    )
    df.sort_values("index").reset_index(drop=True)
    df.index = df["index"]
    df = df.drop(columns=[0, "Month", "index"])
    df = df.sort_index()

    df = frequency_resampling(df)
    df = clean_numeric_columns(df)

    save_processed_file(df, file_name, processed_path)

    return df


# load data from NCEI
def load_ncei(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",
        header=None,
        na_values="-99.99",
        dtype="str",
    )
    codes_df = pd.read_csv("../data/utils_data/PDSI_State_Codes.csv", dtype=str)
    codes_df.index = pd.to_numeric(codes_df["Code"])
    codes_df = codes_df.drop(columns=["Code"])

    df = df.melt(id_vars=[0], var_name="Month", value_name="PDSI")

    df["Year"] = pd.to_numeric(df[0].astype(str).str[-4:])
    df["Region"] = pd.to_numeric(df[0].astype(str).str[:3])
    df["index"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str), format="%Y-%m"
    )
    df.index = df["index"]
    df["PDSI"] = pd.to_numeric(df["PDSI"])
    df = df.drop(columns=[0, "Month", "Year", "index"])

    # As per the 2024 CFPR repository:
    important_states = [4, 13, 25, 41, 11]
    important_regions = [250, 255, 256, 260, 261, 262, 265, 350, 356, 361, 362]

    df = df.loc[df["Region"].isin(important_regions + important_states)]

    # Create individual time series for each regions with the names
    vars_to_include = important_regions + important_states

    dfs_to_include = []

    for var in vars_to_include:
        name = codes_df.loc[codes_df.index == var]["State"].values[0]
        df_var = df[df["Region"] == var]
        df_var = df_var.sort_index()
        df_var = df_var.drop(columns=["Region"])
        df_var.rename(columns={"PDSI": name}, inplace=True)
        dfs_to_include.append(df_var)

    df = pd.concat(dfs_to_include, axis=1)

    df = frequency_resampling(df)
    df = clean_numeric_columns(df)

    save_processed_file(df, file_name, processed_path)

    return df


# load data from CDEC
def load_cdec(file_path, processed_path):
    
    file_name = os.path.basename(file_path)
    df = pd.read_csv(
        file_path, sep=",", engine="python", header=0, na_values="---", dtype="str"
    )
    

    df["index"] = pd.to_datetime(
        df["DATE TIME"].astype(str).str[:4]
        + "-"
        + df["DATE TIME"].astype(str).str[4:6]
        + "-"
        + df["DATE TIME"].astype(str).str[6:8]
    )
    df = df.drop(
        columns=[
            "DURATION",
            "SENSOR_NUMBER",
            "SENSOR_TYPE",
            "DATA_FLAG",
            "UNITS",
            "DATE TIME",
            "OBS DATE",
        ])

    


    df["VALUE"] = pd.to_numeric(df["VALUE"])
    
    #Check to remove any negative values as this means a sensor error 
    df['VALUE'] = df['VALUE'].where(df['VALUE'] >= 0, np.nan)  # Replace negatives with NaN
    
    # Pivot the table by date (index), with STATION_ID as columns
    pivoted_df = df.pivot_table(index='index', columns='STATION_ID', values='VALUE')

    # Take the mean across all STATION_IDs per date
    pivoted_df['SWE'] = pivoted_df.mean(axis=1)

    # Resample to Monthly Start ('MS'), and you can choose how to handle resampling, e.g., using the mean
    df_monthly = pivoted_df.resample('MS').mean()

    df_out = pd.DataFrame(df_monthly['SWE'])


    df = frequency_resampling(df_out)
    df = clean_numeric_columns(df_out)

    save_processed_file(df, file_name, processed_path)

    return df


# load data from BC
def load_bc(file_path, processed_path):
    file_name = os.path.basename(file_path)

    df = pd.read_csv(file_path, index_col=0, skiprows=42)
    df = df.drop(
        columns=[
            "CES_C7_QC",
            "CES_C7_ON",
            "CES_C7_MB",
            "CES_C7_SK",
            "CES_C7_AB",
            "CES_C7_BC",
            "CES_C7_AT",
            "CES_C7_CANADA",
        ]
    )

    # converting quarter dates
    df["Year"] = "Year"
    df["Quarter"] = "Quarter"
    df["index"] = "index"

    df["Year"] = df.index.str[:4]
    df["Quarter"] = df.index.str[4:]

    def convert_to_datetime(row):
        month = int(row["Quarter"][1]) * 3 - 2
        return pd.Timestamp(year=int(row["Year"]), month=month, day=1)

    df["index"] = df.apply(convert_to_datetime, axis=1)

    df.set_index(df["index"], inplace=True)
    df.drop(columns=df.columns[-3:], inplace=True)

    df = df.apply(pd.to_numeric)

    df = frequency_resampling(df)
    df = clean_numeric_columns(df)

    save_processed_file(df, file_name, processed_path)

    return df


# load data from nyfed
def load_nyfed(file_path, processed_path):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, encoding="unicode_escape", skiprows=4)
    df.drop(columns=df.columns[2:], inplace=True)

    df = finalize_df(df)
    df = df.rename(columns={df.columns[0]: file_name[6:-4]})
    save_processed_file(df, file_name, processed_path)

    return df


# Read csv, changes the date, frequency, scale, type and index of data
def process_csv(var_name, processed_path):
    filename = var_name + ".csv"
    file_path = raw_path + var_name + ".csv"
    if filename.startswith("STATSCAN"):
        df = load_statscan(file_path, processed_path)
        return df
    if filename.startswith("IMF"):
        df = load_imf(file_path, processed_path)
        return df
    if filename.startswith("FRED"):
        df = load_fred(file_path, processed_path)
        return df
    if filename.startswith("WB"):
        df = load_world_bank(file_path, processed_path)
        return df
    if filename.startswith("EPU"):
        df = load_Economic_policy_uncertainty(file_path, processed_path)
        return df
    if filename.startswith("GT"):
        df = load_google_trend(file_path, processed_path)
        return df
    if filename.startswith("NOAA"):
        df = load_noaa(file_path, processed_path)
        return df
    if filename.startswith("NCEI"):
        df = load_ncei(file_path, processed_path)
        return df
    if filename.startswith("CDEC"):
        df = load_cdec(file_path, processed_path)
        return df
    if filename.startswith("BC"):
        df = load_bc(file_path, processed_path)
        return df
    if filename.startswith("NYFED"):
        df = load_nyfed(file_path, processed_path)
        return df
