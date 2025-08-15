import pandas as pd
import matplotlib.pyplot as plt
import json


def gb_plot(df):
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    plt.barh(df["feature"], df["coef_magnitude"], color="skyblue")
    plt.xlabel("Magnitude", fontsize=32)
    ax.tick_params(axis="y", labelsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28, ha="right")
    plt.ylabel("Feature", fontsize=32)
    plt.title("Gradient Boost Feature Contribution", fontsize=36)
    plt.savefig(
        "Gradient Boost.png",
        dpi=400,
        bbox_inches="tight",
    )


def rf_plot(df):
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    plt.barh(df["feature"], df["coef_magnitude"], color="skyblue")
    plt.xlabel("Magnitude", fontsize=32)
    ax.tick_params(axis="y", labelsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28, ha="right")
    plt.ylabel("Feature", fontsize=32)
    plt.title("Random Forest Feature Contribution", fontsize=36)
    plt.savefig(
        "Random Forest.png",
        dpi=400,
        bbox_inches="tight",
    )


def hgb_plot(features, importances):
    sorted_indices = sorted(
        range(len(importances)), key=lambda i: importances[i], reverse=True
    )
    features = [features[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]
    plt.figure(figsize=(24, 18))
    plt.barh(features, importances)
    plt.xlabel("magnitude")
    plt.title("Hist Gradient Boost Feature Contribution")
    plt.savefig(
        "Hist Gradient Boost.png",
        dpi=400,
        bbox_inches="tight",
    )


df = pd.read_csv("gb_plot_df.csv")
gb_plot(df)

df = pd.read_csv("rf_plot_df.csv")
rf_plot(df)

with open("hgb_plot_info.json", "r") as f:
    plot_data = json.load(f)
features = plot_data["features"]
importances = plot_data["importances"]
hgb_plot(features, importances)
