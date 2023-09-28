import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.wandb_utils import get_summary, get_run_data
HEIGHT=7
WIDTH=15
PROJECT = "DeepCCA"
# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set(rc={"figure.figsize": (WIDTH, HEIGHT)})
sns.set_style("whitegrid")
sns.set_context(
    "paper",
    font_scale=4.0,
    rc={"lines.linewidth": 2.5, "figure.figsize": (WIDTH, HEIGHT)},
)
MODEL_TO_TITLE = {
    "DCCAEY": "DCCA-EY",
    "DCCANOI": "DCCA-NOI",
    "DCCA": "DCCA-STOL",
}

ORDER = ["DCCA-EY", "DCCA-NOI", "DCCA-STOL"]

os.makedirs("plots/DCCA", exist_ok=True)


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


def plot_minibatch_size_ablation(data="mnist"):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["model"] == "DCCASimpler"]

    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "batch_size"])[f"val/corr"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    best_df = best_df.groupby("batch_size").head(1).reset_index(drop=True)
    # get run data for models in summary_df matching best_df
    summary_df = pd.merge(best_df, summary_df, on=["lr", "batch_size"], how="left")
    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # Change column title _step to samples seen
    df = df.rename(columns={"_step": "Samples Seen"})
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"val/corr": "Validation TCC"})
    sns.lineplot(
        data=df,
        x="epoch",
        y="Validation TCC",
        hue="batch size",
    )
    # x lim 10
    plt.xlim(0, 20)
    plt.ylim(0, 50)
    # tick every 5
    plt.xticks(np.arange(0, 21, 5))
    plt.title(
        "Top 50 DCCA on Split MNIST For DCCA-SVD With\n Different Batch Sizes",
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(f"plots/DCCA/deep_{data}_minibatch_size_ablation.png")


def plot_models_different_batch_sizes(data="SplitMNIST"):
    """
    Get the performance of the best lr for each model at each batch size. On x-axis put batch size, and then grouped bar chart one bar for each model
    """
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] < 500]
    summary_df = summary_df.loc[summary_df["batch_size"] >= 20]

    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "batch_size", "model", "optimizer", "rho"])[f"val/corr"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # Find the best lr, optimizer combination for each batch_size based on train/PCC
    best_lr_per_batch = best_df.groupby(["batch_size", "model"])["val/corr"].idxmax()
    best_lr_df = best_df.loc[best_lr_per_batch][
        ["lr", "batch_size", "model", "optimizer", "rho"]
    ]
    # get summary data for models in summary_df matching best_lr_df drop duplicate columns
    summary_df = pd.merge(
        best_lr_df,
        summary_df,
        on=["lr", "batch_size", "model", "optimizer", "rho"],
        how="left",
    )
    # Use formal model names
    summary_df["model"] = summary_df["model"].map(MODEL_TO_TITLE)
    summary_df = summary_df.rename(columns={"batch_size": "batch size"})
    summary_df = summary_df.rename(columns={"val/corr": "Validation TCC"})
    # as grouped bar chart, x-axis is batch size, y-axis is train/PCC, grouped by model
    # Set the figure size to make it shorter and wider
    g = sns.catplot(
        data=summary_df,
        x="batch size",
        y="Validation TCC",
        hue="model",
        hue_order=ORDER,
        kind="bar",
        palette=colorblind_palette,
        height=HEIGHT,
        aspect=WIDTH/HEIGHT,
    )
    plt.title(rf"Top 50 CCA on {data}")
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
    plt.tight_layout()
    plt.savefig(
        f"plots/DCCA/{data}_models_different_batch_sizes.svg", bbox_inches="tight"
    )
    plt.close()


def plot_all_learning_curves(data="SplitMNIST", batch_sizes=None):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    df = pd.DataFrame()  # Initialize an empty DataFrame to hold all the data

    for batch_size in batch_sizes:
        single_df = get_best_runs(data=data, batch_size=batch_size)
        df = pd.concat(
            [df, single_df], ignore_index=True
        )  # Concatenate to the main DataFrame
    df = df.rename(columns={"val/corr": "Validation TCC"})
    # drop nans in Validation TCC
    df = df.dropna(subset=["Validation TCC"])
    df = df.rename(columns={"train/corr": "Train TCC"})
    df = df.rename(columns={"batch_size": "batch size"})
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    # Plot the learning curves
    g = sns.lineplot(
        data=df,
        x="epoch",
        y="Validation TCC",
        hue="model",
        hue_order=ORDER,
        palette=colorblind_palette,
        style="batch size",
        style_order=batch_sizes,
    )
    plt.legend(fontsize="x-small")
    plt.title(rf"Top 50 CCA on {data}")
    # sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=4, title=None)
    # sns.move_legend(g, "center left", bbox_to_anchor=(1, 0.5), ncol=1, title=None, frameon=False)
    plt.tight_layout()
    plt.savefig(f"plots/DCCA/{data}_allbatchsizes_pcc.svg")
    plt.close()


def main():
    plot_models_different_batch_sizes(data="SplitMNIST")
    plot_models_different_batch_sizes(data="XRMB")
    plot_all_learning_curves(data="SplitMNIST", batch_sizes=[50, 100])
    plot_all_learning_curves(data="XRMB", batch_sizes=[50, 100])


if __name__ == "__main__":
    main()
