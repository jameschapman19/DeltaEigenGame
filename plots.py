"""
Generates plots of results from wandb api
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import numpy as np

from wandb_utils import get_summary, get_run_data

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

PROJECT = "DeltaEigenGame"

MODEL_TO_TITLE = {
    "delta": r"kGHGEP",
    "sgha": "SGHA",
    "gamma": r"$\gamma$" + "-EigenGame",
    "sp": "Stochastic Power",
    "ghgep": "GHGEP",
    "eygep": "EYGEP",
}

# Set order of models in plots
ORDER = [
    "EYGEP",
    "GHGEP",
    "SGHA",
    r"$\gamma$" + "-EigenGame",
    "Stochastic Power",
]


DIMENSIONS = {
    "mnist": (392, 392),
    "cifar": (1536, 1536),
    "mediamill": (120, 120),
}


def get_best_runs(
    data="mnist", batch_size=100, objective="PCC", mode="Train", momentum=0.9, lr=None
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    summary_df = summary_df.loc[summary_df["momentum"] == momentum]
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
    if objective == "PCC":
        summary_df = summary_df.loc[summary_df["objective"] == "cca"]
    elif objective == "PVC":
        summary_df = summary_df.loc[summary_df["objective"] == "pls"]
    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["model", "lr", "momentum"])[f"{mode} {objective}"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # summary_df = summary_df.loc[summary_df['momentum'] == momentum]
    # sort summary_df by Train PCC or PVC
    best_df = best_df.sort_values(by=[f"{mode} {objective}"], ascending=False)
    best_df = best_df.groupby("model").head(1).reset_index(drop=True)
    # get run data for models in summary_df matching best_df
    summary_df = pd.merge(
        best_df, summary_df, on=["model", "lr", "momentum"], how="left"
    )
    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # Change column title _step to samples seen
    df = df.rename(columns={"_step": "Samples Seen"})
    return df


def plot_pcc(data="mnist", batch_size=100, momentum=0.9, lr=None):
    # Plot PCC for best runs for each model
    df = get_best_runs(
        data=data,
        batch_size=batch_size,
        objective="PCC",
        mode="Train",
        momentum=momentum,
        lr=lr,
    )
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    plt.figure()
    sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="model",
        hue_order=ORDER[:-1],
    )
    plt.title(
        rf"Top 4 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    if lr is None:
        lr = "tuned"
    plt.savefig(f"plots/{data}_{batch_size}_pcc_lr_{lr}.png")


def plot_pvc(data="mnist", batch_size=100, momentum=0.9, lr=None):
    # Plot PVC for best runs for each model
    df = get_best_runs(
        data=data,
        batch_size=batch_size,
        objective="PVC",
        mode="Train",
        momentum=momentum,
        lr=lr,
    )
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    plt.figure()
    sns.lineplot(data=df, x="Samples Seen", y="Train PVC", hue="model", hue_order=ORDER)
    plt.title(
        rf"Top 4 PLS on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    if lr is None:
        lr = "tuned"
    plt.savefig(f"plots/{data}_{batch_size}_pvc_lr_{lr}.png")


for data in ["mnist", "cifar", "mediamill"]:
    for batch_size in [100]:
        for lr in [0.001, 0.01]:
            # plot_pvc(data=data, batch_size=batch_size, momentum=0, lr=lr)
            plot_pcc(data=data, batch_size=batch_size, momentum=0.5, lr=lr)
