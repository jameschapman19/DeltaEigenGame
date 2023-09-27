import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.wandb_utils import get_summary, get_run_data

PROJECT = "DeepCCA"
# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set_context("paper", font_scale=2.0)
sns.set_style("whitegrid")
# Set the default figure size
plt.rcParams["figure.figsize"] = (24, 6)  # Adjust the values as needed
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


def plot_all_models(data="XRMB", lr=None, batch_size=100):
    df = get_best_runs(data=data, lr=lr, batch_size=batch_size)
    plot_tcc(df, title=f"dcca_{data}", data=data, hue="model")


def plot_tcc(run_data, title, data="XRMB", hue="model"):
    # get only the models in MODEL_TO_TITLE
    run_data = run_data.loc[run_data["model"].isin(MODEL_TO_TITLE.keys())]
    # map model names to titles
    run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
    # rename val/corr to Validation TCC
    run_data = run_data.rename(columns={"val/corr": "Validation TCC"})
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=run_data,
        x="epoch",
        y="Validation TCC",
        hue=hue,
        hue_order=ORDER,
    )
    # x lim 10
    plt.xlim(0, 20)
    plt.ylim(0, 50)
    # tick every 5
    plt.xticks(np.arange(0, 21, 5))
    plt.title(rf"Top 50 DCCA on {data}")
    plt.tight_layout()
    plt.savefig(f"plots/DCCA/{title}.png")


def plot_simpler_lr():
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == "SplitMNIST"]
    # batch size 100
    summary_df = summary_df.loc[summary_df["batch_size"] == 100]
    # For DCCAEY get all runs
    summary_df = summary_df.loc[summary_df["model"] == "DCCASimpler"]
    run_data = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # map model names to titles
    run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
    # rename val/corr to Validation TCC
    run_data = run_data.rename(columns={"val/corr": "Validation TCC"})
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=run_data,
        x="epoch",
        y="Validation TCC",
        hue="lr",
        hue_order=ORDER,
    )

    plt.tight_layout()
    # x lim 10
    plt.xlim(0, 20)
    plt.ylim(0, 50)
    # tick every 5
    plt.xticks(np.arange(0, 21, 5))
    plt.title(
        "Top 50 DCCA on Split MNIST For DCCA-SVD With Different Learning Rates",
        wrap=True,
    )
    plt.savefig(f"plots/DCCA/dcca_lr_experiment.png")


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
    plt.figure(figsize=(10, 5))
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
    summary_df = summary_df.loc[summary_df["batch_size"] <500]

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
    g=sns.catplot(
        data=summary_df,
        x="batch size",
        y="Validation TCC",
        hue="model",
        hue_order=ORDER,
        kind="bar",
        palette=colorblind_palette,
    )
    g.fig.set_figwidth(11.87)
    g.fig.set_figheight(8.27)
    plt.title(rf"Top 50 CCA on {data}")
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
    plt.tight_layout()
    plt.savefig(f"plots/DCCA/{data}_models_different_batch_sizes.svg", bbox_inches="tight")


def main():
    plot_models_different_batch_sizes(data="XRMB")
    plot_models_different_batch_sizes(data="SplitMNIST")


if __name__ == "__main__":
    main()
