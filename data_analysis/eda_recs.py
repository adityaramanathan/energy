import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Functions for EDA


def statistics(df, feature):
    print(feature)
    median = round(df[feature].median(), 2)
    print(median)
    std = round(df[feature].std(), 2)
    print(std)
    min = round(df[feature].min(), 2)
    print(min)
    max = round(df[feature].max(), 2)
    print(max)
    print("\n")

    new_row = pd.DataFrame(
        {
            "Feature": [feature],
            "Median": [median],
            "Standard Deviation": [std],
            "Min": [min],
            "Max": [max],
        }
    )

    global stats_df
    if stats_df.empty:
        stats_df = new_row
    else:
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)


def energy_vs_feature(df, feature, energy, descriptor):
    p = np.polyfit(df[feature], df[energy], 1)
    f = np.poly1d(p)

    x_new = df[feature].values
    y_new = f(x_new)

    plt.figure(figsize=png_size)
    plt.scatter(
        df[feature],
        df[energy],
        marker="o",
        label="Data Points",
    )
    plt.plot(x_new, y_new, color="red", label="Fit Line", linestyle="-")
    plt.xlabel(feature, fontsize=24)
    plt.xticks(fontsize=18)
    plt.ylabel(descriptor, fontsize=24)
    plt.yticks(fontsize=18)
    plt.title(descriptor + " vs. " + feature, fontsize=30)
    plt.legend()
    plt.savefig(
        os.path.join(
            "visualizations_" + descriptor, feature + " " + energy + " Scatter.png"
        ),
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()


def mapping_to_energy_type(df, category):
    counts = {
        category: df[category].unique(),
        "Electricity": df[df["TOTAL_ELECTRICITY_USE_1000BTU"] != 0]
        .groupby(category)
        .size(),
        "Gas": df[df["TOTAL_GAS_USE_1000BTU"] != 0].groupby(category).size(),
        "Propane": df[df["TOTAL_PROPANE_USE_1000BTU"] != 0].groupby(category).size(),
        "Oil/Kerosene": df[df["TOTAL_OILKEROSENE_USE_1000BTU"] != 0]
        .groupby(category)
        .size(),
        "Wood": df[df["TOTAL_WOOD_USE_1000BTU"] != 0].groupby(category).size(),
    }

    counts_df = pd.DataFrame(counts)

    fig, ax = plt.subplots(figsize=(24, 12))
    counts_df.plot(kind="bar", ax=ax)

    ax.set_ylabel("Number of Households", fontsize=20)
    ax.set_title("Number of Households by Energy Source and " + category, fontsize=30)
    ax.legend(title="Energy Source")

    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    plt.savefig(
        os.path.join("visualizations", "Mapping " + category + " to Energy Type.png"),
        dpi=400,
        bbox_inches="tight",
    )


# 1 time use functions


def sort_year_category(series):
    return series.sort_index(key=lambda x: x.str[:4])


def calculate_center_date(date_range):
    if "Before" in date_range:
        return int(date_range[7:])
    else:
        parts = date_range.split(" - ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) / 2


def calculate_center_income(income_range):
    if "Less than" in income_range:
        return int(income_range[10:]) / 2
    elif income_range[0:6] == "150000":
        return 150000
    else:
        parts = income_range.split(" - ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) / 2


# main

df = pd.read_csv("recs_filtered_2020_eda.csv")
df_all_energies = pd.read_csv("recs_filtered_2020_eda_with_all_energies.csv")
os.makedirs("visualizations", exist_ok=True)
os.makedirs("visualizations_energy", exist_ok=True)
os.makedirs("visualizations_electricity", exist_ok=True)
os.makedirs("visualizations_gas", exist_ok=True)
png_size = (24, 12)

# data changes for more clear visualizaations

df["YEARMADE"] = df["YEARMADERANGE"].apply(calculate_center_date)
df["HOUSEINCOME"] = df["ANNUAL_GROSS_HOUSE_INCOME"].apply(calculate_center_income)

df = df[df["TOTAL_ELECTRICITY_USE_1000BTU"] <= 300000]
df = df[df["TOTAL_GAS_USE_1000BTU"] <= 500000]

# Summary Statistics

stats_df = pd.DataFrame(
    columns=["Feature", "Central Tendency", "Standard Deviation", "Min", "Max"]
)

statistics(df, "TOTAL_ENERGY_1000BTU")
statistics(df, "REPORTED_SQ_FT")
statistics(df, "NUM_TOTAL_ROOMS")
statistics(df, "HOUSE_MEMBER_COUNT")
statistics(df, "TOTAL_ELECTRICITY_COST_$")
statistics(df, "TOTAL_NATURAL_GAS_COST_$")
statistics(df, "NUM_BEDROOMS")
statistics(df, "NUM_FULL_BATHROOMS")
statistics(df, "NUM_HALF_BATHROOMS")
statistics(df, "NUM_OTHER_ROOMS")
statistics(df, "NUM_WEEKDAYS_AT_HOME")
statistics(df, "NUMFRIG")
statistics(df, "NUMFREEZ")
statistics(df, "NUMOVEN")
statistics(df, "NUMMICRO")
statistics(df, "NUM_ELECTRONIC_DEVICES")
statistics(df, "NUMCFAN")
statistics(df, "NUMLIGHTS_1_4_HRS_DAY")
statistics(df, "NUMLIGHTS_4_8_HRS_DAY")
statistics(df, "NUMLIGHTS_MORE_8_HRS_DAY")
statistics(df, "HDD")
statistics(df, "CDD")

stats_df.to_csv("statistics.csv")

# All Energy Types Distributions

energy_columns = [
    "TOTAL_ELECTRICITY_USE_1000BTU",
    "TOTAL_GAS_USE_1000BTU",
    "TOTAL_PROPANE_USE_1000BTU",
    "TOTAL_OILKEROSENE_USE_1000BTU",
    "TOTAL_WOOD_USE_1000BTU",
]
non_zero_counts = df_all_energies[energy_columns].apply(lambda x: (x != 0).sum())
plt.figure(figsize=(24, 12))
plt.bar(non_zero_counts.index, non_zero_counts.values, color="skyblue")
plt.title("House Count per Energy Type", fontsize=30)
plt.xlabel("Energy Type", fontsize=24)
plt.ylabel("Number of Houses using that Energy Type", fontsize=24)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(
    os.path.join("Visualizations", "House Count per Energy Type.png"),
    dpi=400,
    bbox_inches="tight",
)
plt.close()

# Bar chart showing total energy per state

df_all_energies["TOTAL_ENERGY_1000BTU"] = df_all_energies[
    [
        "TOTAL_ELECTRICITY_USE_1000BTU",
        "TOTAL_GAS_USE_1000BTU",
        "TOTAL_PROPANE_USE_1000BTU",
        "TOTAL_OILKEROSENE_USE_1000BTU",
        "TOTAL_WOOD_USE_1000BTU",
    ]
].sum(axis=1)

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
}

energy_per_state = (
    df_all_energies.groupby("state_name")["TOTAL_ENERGY_1000BTU"]
    .mean()
    .sort_values(ascending=False)
)
plt.figure(figsize=(24, 12))

top10 = energy_per_state.head(10)
states = [state_abbreviations[state] for state in top10.index]
energy_vals = top10.values

plt.bar(states, energy_vals, color="forestgreen")
plt.xlabel("State", fontsize=24)
plt.ylabel("Total Energy Used in 1000BTU", fontsize=24)
plt.title("Average Household Energy Use per State", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = plt.gca()
ax.yaxis.get_offset_text().set_fontsize(20)
plt.savefig(
    os.path.join("Visualizations", "Total site consumption"),
    dpi=400,
    bbox_inches="tight",
)
plt.close()

# Energy - Electricity + Gas

energy_vs_feature(df, "HOUSEINCOME", "TOTAL_ENERGY_1000BTU", "energy")
energy_vs_feature(df, "TOTAL_ELECTRICITY_COST_$", "TOTAL_ENERGY_1000BTU", "energy")
energy_vs_feature(df, "TOTAL_NATURAL_GAS_COST_$", "TOTAL_ENERGY_1000BTU", "energy")
energy_vs_feature(df, "HDD", "TOTAL_ENERGY_1000BTU", "energy")
energy_vs_feature(df, "CDD", "TOTAL_ENERGY_1000BTU", "energy")


# Electricity Relationships

energy_vs_feature(df, "HOUSEINCOME", "TOTAL_ELECTRICITY_USE_1000BTU", "electricity")
energy_vs_feature(
    df, "TOTAL_ELECTRICITY_COST_$", "TOTAL_ELECTRICITY_USE_1000BTU", "electricity"
)
energy_vs_feature(
    df, "TOTAL_NATURAL_GAS_COST_$", "TOTAL_ELECTRICITY_USE_1000BTU", "electricity"
)
energy_vs_feature(df, "HDD", "TOTAL_ELECTRICITY_USE_1000BTU", "electricity")
energy_vs_feature(df, "CDD", "TOTAL_ELECTRICITY_USE_1000BTU", "electricity")


# Gas Relationships

energy_vs_feature(df, "HOUSEINCOME", "TOTAL_GAS_USE_1000BTU", "gas")
energy_vs_feature(df, "TOTAL_ELECTRICITY_COST_$", "TOTAL_GAS_USE_1000BTU", "gas")
energy_vs_feature(df, "TOTAL_NATURAL_GAS_COST_$", "TOTAL_GAS_USE_1000BTU", "gas")
energy_vs_feature(df, "HDD", "TOTAL_GAS_USE_1000BTU", "gas")
energy_vs_feature(df, "CDD", "TOTAL_GAS_USE_1000BTU", "gas")


# Mapping a Feature to the Type of Energy

mapping_to_energy_type(df_all_energies, "YEARMADERANGE")
mapping_to_energy_type(df_all_energies, "HOUSEHOLDER_RACE")
mapping_to_energy_type(df_all_energies, "DIVISION")
