"""
Generates plots of results from wandb api
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.wandb_utils import get_summary, get_run_data

# Set a consistent color scheme for NeurIPS paper
colors = sns.color_palette("colorblind")
sns.set_palette(colors)
continuous_palette = sns.color_palette("cividis", as_cmap=True)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5, "axes.labelsize": 16})
# sns tight layout

PROJECT = "StochasticCCA"

MODEL_TO_TITLE = {
    "ey": "EY",
    "gha": "SGHA",
    "gamma": r"$\gamma$" + "-EigenGame",
}

# Set order of models in plots
ORDER = [
    "EY",
    "SGHA",
    r"$\gamma$" + "-EigenGame",
]

DIMENSIONS = {
    "mnist": (392, 392),
    "cifar": (1536, 1536),
    "mediamill": (120, 120),
}


def get_best_runs(
        data="mnist", batch_size=100, objective="PCC", mode="Train", lr=None, seed=None
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
    if seed is not None:
        summary_df = summary_df.loc[summary_df["seed"] == seed]
    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["model", "lr", "optimizer"])[f"{mode.lower()}/{objective}"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # sort summary_df by Train PCC or PVC
    best_df = best_df.sort_values(by=[f"{mode.lower()}/{objective}"], ascending=False)
    best_df = best_df.groupby("model").head(1).reset_index(drop=True)
    # get run data for models in summary_df matching best_df
    summary_df = pd.merge(
        best_df, summary_df, on=["model", "lr", "optimizer"], how="left"
    )
    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    return df


def plot_pcc(data="mnist", batch_size=100, lr=None, time=False):
    if time:
        seed = 1
    else:
        seed = None
    # Plot PCC for best runs for each model
    df = get_best_runs(
        data=data,
        batch_size=batch_size,
        objective="PCC",
        mode="Train",
        lr=lr,
        seed=seed
    )
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"train/PCC": "Train PCC"})
    df = df.rename(columns={"samples_seen": "Samples Seen"})
    df = df.rename(columns={"_runtime": "Time"})
    # Fill NaN values in "Samples Seen" column with previous values
    df["Samples Seen"].fillna(method="ffill", inplace=True)
    # drop rows with Nan in "Train PCC" column
    df = df.dropna(subset=["Train PCC"])
    # figure that is shorter than it is wide
    plt.figure(figsize=(10, 5))
    if time:
        sns.lineplot(
            data=df,
            x="Time",
            y="Train PCC",
            hue="model",
            hue_order=ORDER,
        )
    else:
        sns.lineplot(
            data=df,
            x="Samples Seen",
            y="Train PCC",
            hue="model",
            hue_order=ORDER,
        )
    plt.title(
        rf"Top 4 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    if lr is None:
        lr = "tuned"
    plt.ylim(0, 1)
    plt.tight_layout()
    if time:
        plt.savefig(f"plots/{data}_{batch_size}_pcc_time_lr_{lr}.svg")
    plt.savefig(f"plots/{data}_{batch_size}_pcc_lr_{lr}.svg")


def plot_minibatch_size_ablation(data="mnist", optimizer="Adam", time=False):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["model"] == "ey"]
    summary_df = summary_df.loc[summary_df["optimizer"] == optimizer]

    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "batch_size"])[f"train/PCC"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # Find the best lr for each batch_size based on train/PCC
    best_lr_per_batch = best_df.groupby("batch_size")["train/PCC"].idxmax()
    best_lr_df = best_df.loc[best_lr_per_batch]
    # get run data for models in summary_df matching best_lr_df
    summary_df = pd.merge(
        best_lr_df, summary_df, on=["lr", "batch_size"], how="left"
    )
    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"train/PCC": "Train PCC"})
    df = df.rename(columns={"samples_seen": "Samples Seen"})
    # Fill NaN values in "Samples Seen" column with previous values
    df["Samples Seen"].fillna(method="ffill", inplace=True)
    # drop rows with Nan in "Train PCC" column
    df = df.dropna(subset=["Train PCC"])
    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="batch size",
        # hue color palette
        palette=continuous_palette,
    )
    plt.ylim(0, 1)
    plt.title(
        rf"Top 4 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    plt.savefig(f"plots/{data}_minibatch_size_ablation.svg")

def plot_learning_rate_ablation(
    data="mnist", batch_size=100, optimizer="Adam"
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    summary_df = summary_df.loc[summary_df["model"] == "ey"]
    summary_df = summary_df.loc[summary_df["optimizer"] == optimizer]


    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"train/PCC": "Train PCC"})
    df = df.rename(columns={"samples_seen": "Samples Seen"})
    # Fill NaN values in "Samples Seen" column with previous values
    df["Samples Seen"].fillna(method="ffill", inplace=True)
    # drop rows with Nan in "Train PCC" column
    df = df.dropna(subset=["Train PCC"])
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="lr",
        # hue color palette
        palette=continuous_palette,
    )
    plt.ylim(0, 1)

if __name__ == '__main__':
    # plot_minibatch_size_ablation("mediamill")
    # plot_minibatch_size_ablation("cifar")
    for data in ["mediamill","cifar"]:
        for batch_size in [100, 50, 20, 5]:
            plot_pcc(data, batch_size, time=True)

