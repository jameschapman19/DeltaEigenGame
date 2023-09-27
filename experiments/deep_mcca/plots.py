import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.wandb_utils import get_summary, get_run_data

# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set_style("whitegrid")
sns.set_context(
    "paper", font_scale=2.0, rc={"lines.linewidth": 2.5, "axes.labelsize": 16}
)
# sns tight layout
# Set the default figure size
plt.rcParams["figure.figsize"] = (18, 6)  # Adjust the values as needed
PROJECT = "DeepMCCA"

MODEL_TO_TITLE = {
    "DCCAEY": "DCCA-EY",
    "DMCCA": "DMCCA",
    "DGCCA": "DGCCA",
}

ORDER = ["DCCA-EY", "DMCCA", "DGCCA"]

os.makedirs("plots/DMCCA", exist_ok=True)


def get_best_runs(
    data="SplitMNIST",
    lr=None,
    batch_size=100,
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    best_df = summary_df.sort_values(by=[f"val/corr"], ascending=False)
    best_df = best_df.groupby("model").head(1).reset_index(drop=True)
    df = get_run_data(ids=best_df["id"].tolist(), project=PROJECT)
    return df


def plot_learning_curves(data="mfeat", batch_size=200):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    # get the run data for batch size
    df = get_best_runs(data=data, batch_size=batch_size)
    #drop nans in Validation TMCC
    df = df.rename(columns={"val/corr": "Validation TMCC"})
    df = df.rename(columns={"train/corr": "Train TMCC"})
    df = df.rename(columns={"batch_size": "batch size"})
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    # Plot the learning curves
    plt.figure(figsize=(12, 6))
    g = sns.lineplot(
        data=df,
        x="epoch",
        y="Validation TMCC",
        hue="model",
        hue_order=ORDER,
        palette=colorblind_palette,
    )
    plt.title(rf"Top 50 CCA on {data}")
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig(f"plots/DMCCA/{data}_{batch_size}_pcc.svg")

def plot_all_learning_curves(data="mfeat", batch_sizes=None):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    df = pd.DataFrame()  # Initialize an empty DataFrame to hold all the data

    for batch_size in batch_sizes:
        single_df = get_best_runs(data=data, batch_size=batch_size)
        df = pd.concat([df, single_df], ignore_index=True)  # Concatenate to the main DataFrame
    #drop nans in Validation TMCC
    df = df.rename(columns={"val/corr": "Validation TMCC"})
    df = df.rename(columns={"train/corr": "Train TMCC"})
    df = df.rename(columns={"batch_size": "batch size"})
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    # Plot the learning curves
    plt.figure(figsize=(12, 6))
    g = sns.lineplot(
        data=df,
        x="epoch",
        y="Validation TMCC",
        hue="model",
        hue_order=ORDER,
        palette=colorblind_palette,
        style="batch size",
        style_order=batch_sizes,
    )
    plt.title(rf"Top 50 CCA on {data}")
    #plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig(f"plots/DMCCA/{data}_allbatchsizes_pcc.svg")



def plot_models_different_batch_sizes(data="mfeat"):
    """
    Get the performance of the best lr for each model at each batch size. On x-axis put batch size, and then grouped bar chart one bar for each model
    """
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]

    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "batch_size", "model", "optimizer"])[f"val/corr"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # Find the best lr, optimizer combination for each batch_size based on train/PCC
    best_lr_per_batch = best_df.groupby(["batch_size", "model"])["val/corr"].idxmax()
    best_lr_df = best_df.loc[best_lr_per_batch][
        ["lr", "batch_size", "model", "optimizer"]
    ]
    # get summary data for models in summary_df matching best_lr_df drop duplicate columns
    summary_df = pd.merge(
        best_lr_df,
        summary_df,
        on=["lr", "batch_size", "model", "optimizer"],
        how="left",
    )
    # Use formal model names
    summary_df["model"] = summary_df["model"].map(MODEL_TO_TITLE)
    summary_df = summary_df.rename(columns={"batch_size": "batch size"})
    summary_df = summary_df.rename(columns={"val/corr": "Validation TCC"})
    # as grouped bar chart, x-axis is batch size, y-axis is train/PCC, grouped by model
    plt.figure(figsize=(12, 6))
    g = sns.catplot(
        data=summary_df,
        x="batch size",
        y="Validation TCC",
        hue="model",
        hue_order=ORDER,
        kind="bar",
        palette=colorblind_palette,
    )
    plt.title(rf"Top 50 CCA on {data}")
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
    plt.tight_layout()
    plt.savefig(f"plots/DMCCA/{data}_models_different_batch_sizes.svg", bbox_inches="tight")


def main():
    plot_all_learning_curves(data="mfeat", batch_sizes=[50, 200])
    for batch_size in [5, 10,20, 50, 100, 200]:
        plot_learning_curves(data="mfeat", batch_size=batch_size)
    plot_models_different_batch_sizes()


if __name__ == "__main__":
    main()
