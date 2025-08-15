# BEFORE RUNNING THIS FILE with NEW CHANGES, the data_simplification.py file MUST be run before models can be used!
import pandas as pd
from sklearn.model_selection import train_test_split


def encode(df, column_name):
    df = pd.get_dummies(df, columns=[column_name])
    return df


def preprocess():
    df = pd.read_csv("recs_filtered_2020.csv")

    # One-hot encoding of the categorical non-ordinal columns

    needs_encoding = [
        "HOUSING_TYPE",
        "EMPLOYMENT_STATUS",
        "HOUSEHOLDER_RACE",
        "DIVISION",
        "BA_climate",
    ]

    for category in needs_encoding:
        df = encode(df, category)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Prepare for Correlation Check

    columns = [
        "TOTAL_ELECTRICITY_USE_KILOWATTHRS",
        "TOTAL_ELECTRICITY_USE_1000BTU",
        "TOTAL_ELECTRICITY_COST_$",
        "CONVERSION_ELECTRICITY",
        "TOTAL_GAS_USE_100ft3",
        "TOTAL_GAS_USE_1000BTU",
        "TOTAL_NATURAL_GAS_COST_$",
        "CONVERSION_GAS",
        "TOTAL_PROPANE_USE_GALLONS",
        "TOTAL_PROPANE_USE_1000BTU",
        "TOTAL_PROPANE_COST_$",
        "CONVERSION_PROPANE",
        "TOTAL_OILKEROSENE_USE_GALLONS",
        "TOTAL_OILKEROSENE_USE_1000BTU",
        "TOTAL_OILKEROSENE_COST_$",
        "CONVERSION_OILKEROSENE",
        "TOTAL_WOOD_USE_1000BTU",
        "TOTAL_ENERGY_1000BTU",
    ]

    df_portion = df[columns]

    col = df["state_name"]
    columns.append("state_name")

    # REPLACE THIS CODE TO REMOVE THE STATE NAME COLUMN

    df = df.drop(columns=columns)

    # Removing highly-correlated variables
    print(
        "Column Count before Correlation Removal: "
        + str(len(df.columns) + len(columns))
    )

    corr_matrix = df.corr()
    thresh = 0.75
    to_drop = set()

    correlated_columns = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > thresh:
                correlated_columns.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j])
                )

    print(correlated_columns)

    # Due to those results, decided to remove NUM_TOTAL_ROOMS

    to_drop.add("NUM_TOTAL_ROOMS")

    df = df.drop(columns=to_drop)

    # Undo preparation changes

    df.insert(0, "state_name", col)
    df = pd.concat([df, df_portion], axis=1)

    # Finish

    print("Column Count Correlation Removal: " + str(len(df.columns)))
    print("The length of the dataframe is " + str(len(df)))
    df.to_csv("recs_filtered_2020.csv", index=False)


def input_data_to_model():
    df = pd.read_csv("recs_filtered_2020.csv")
    df = df[df["TOTAL_ENERGY_1000BTU"] <= 500000]  # outlier removal

    features = [
        # "RENT_OR_OWN",
        "HDD",
        "CDD",
        "YEARMADERANGE",
        "NUM_BEDROOMS",
        "NUM_FULL_BATHROOMS",
        "NUM_HALF_BATHROOMS",
        "NUM_OTHER_ROOMS",
        "NUMFRIG",
        "NUMFREEZ",
        "NUMOVEN",
        "NUMMICRO",
        "NUM_ELECTRONIC_DEVICES",
        "NUMCFAN",
        "NUMLIGHTS_1_4_HRS_DAY",
        "NUMLIGHTS_4_8_HRS_DAY",
        "NUMLIGHTS_MORE_8_HRS_DAY",
        "EDUCATION",
        # "HISPANIC/LATINO_OR_NOT",
        "HOUSE_MEMBER_COUNT",
        # "NUM_WEEKDAYS_AT_HOME",
        "ANNUAL_GROSS_HOUSE_INCOME",
        "REPORTED_SQ_FT",
        "TOTAL_ELECTRICITY_COST_$",
        "TOTAL_NATURAL_GAS_COST_$",
        # "HOUSING_TYPE_Apartment in building with 2-4 units",
        "HOUSING_TYPE_Apartment in building with >= 5 units",
        # "HOUSING_TYPE_Mobile Home",
        # "HOUSING_TYPE_Single-Family Attached",
        "HOUSING_TYPE_Single-Family Detached",
        # "EMPLOYMENT_STATUS_Full-time",
        # "EMPLOYMENT_STATUS_Not employed",
        # "EMPLOYMENT_STATUS_Part-time",
        # "EMPLOYMENT_STATUS_Retired",
        # "HOUSEHOLDER_RACE_>= 2 Races",
        # "HOUSEHOLDER_RACE_African American",
        # "HOUSEHOLDER_RACE_American Indian/Natives",
        # "HOUSEHOLDER_RACE_Asian",
        # "HOUSEHOLDER_RACE_Native Hawaiian/Pacific Islander",
        # "HOUSEHOLDER_RACE_White",
        "DIVISION_East North Central",
        "DIVISION_East South Central",
        "DIVISION_Mountain North",
        # "DIVISION_Mountain South",
        "DIVISION_New England",
        "DIVISION_Pacific",
        "DIVISION_South Atlantic",
        "DIVISION_West North Central",
        "DIVISION_West South Central",
        "BA_climate_Cold",
        "BA_climate_Hot-Dry",
        # "BA_climate_Hot-Humid",
        # "BA_climate_Marine",
        # "BA_climate_Mixed-Dry",
        "BA_climate_Mixed-Humid",
        # "BA_climate_Very-Cold",
    ]

    X = df[features]

    y = df["TOTAL_ENERGY_1000BTU"]  # Energy = Electricity + Gas

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    return X_train, y_train, X_test, y_test, features


if __name__ == "__main__":
    preprocess()
