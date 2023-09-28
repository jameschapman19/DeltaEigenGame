"""
Generates plots of results from wandb api
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.wandb_utils import get_summary, get_run_data
HEIGHT = 7
WIDTH = 15
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

os.makedirs("plots/StochasticCCA", exist_ok=True)


def get_best_runs(
    data="mnist",
    batch_size=100,
    objective="PCC",
    mode="Train",
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["model", "lr", "optimizer", "gamma"])[f"{mode.lower()}/{objective}"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # sort summary_df by Train PCC or PVC
    best_df = best_df.sort_values(by=[f"{mode.lower()}/{objective}"], ascending=False)
    best_df = best_df.groupby(["model"]).head(1).reset_index(drop=True)

    # get run data for models in summary_df matching best_df
    summary_df = pd.merge(
        best_df, summary_df, on=["model", "lr", "optimizer", "gamma"], how="left"
    )
    df = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    return df


def plot_all_learning_curves(data="mnist", batch_sizes=None):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    df = pd.DataFrame()  # Initialize an empty DataFrame to hold all the data
    for batch_size in batch_sizes:
        single_df = get_best_runs(data=data, batch_size=batch_size)
        df = pd.concat(
            [df, single_df], ignore_index=True
        )  # Concatenate to the main DataFrame
    # keep rows where train/penaties_epoch is Nan
    df = df.loc[df["train/penalties_epoch"].isna()]
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"train/PCC": "Train PCC"})
    df = df.rename(columns={"samples_seen": "Samples Seen"})
    mask_gamma = df["model"] == "gamma"
    # Fill NaN values in "Train PCC" column with forward fill for non-"gamma" models
    df.loc[~mask_gamma, "Train PCC"] = df.loc[~mask_gamma, "Train PCC"].fillna(method="ffill")

    # Fill NaN values in "Train PCC" column with backward fill for "gamma" models
    df.loc[mask_gamma, "Train PCC"] = df.loc[mask_gamma, "Train PCC"].fillna(method="bfill")
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    # drop rows with Nan in "Train PCC" column
    df = df.dropna(subset=["Samples Seen"])
    df = df[["model", "batch size", "Samples Seen", "Train PCC"]]
    g = sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="model",
        hue_order=ORDER,
        palette=sns.color_palette(palette, n_colors=3),
        style="batch size",
        style_order=batch_sizes,

    )
    plt.legend(fontsize="x-small")
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    plt.ylim(0, 1)
    if data=="mediamill":
        plt.xlim(0, 25000)
    else:
        plt.xlim(0, 50000)
    plt.tight_layout()
    plt.savefig(
        f"plots/StochasticCCA/{data}_allbatchsizes_pcc.svg"
    )
    plt.close()


def plot_optimizer_ablation(data="mnist", batch_size=100):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    summary_df = summary_df.loc[summary_df["model"] == "ey"]
    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "optimizer"])[f"train/PCC"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # Find the best lr for each optimizer based on train/PCC
    best_lr_per_batch = best_df.groupby("optimizer")["train/PCC"].idxmax()
    best_lr_df = best_df.loc[best_lr_per_batch]
    # get run data for models in summary_df matching best_lr_df
    summary_df = pd.merge(best_lr_df, summary_df, on=["lr", "optimizer"], how="left")
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
    sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        style="optimizer",
        palette=colorblind_palette,
    )
    plt.ylim(0, 1)
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    plt.savefig(f"plots/StochasticCCA/{data}_optimizer_ablation.svg")
    plt.close()


def plot_models_different_batch_sizes(data="mnist"):
    """
    Get the performance of the best lr for each model at each batch size. On x-axis put batch size, and then grouped bar chart one bar for each model
    """
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] > 2]

    # get average over random seeds
    best_df = (
        summary_df.fillna(np.inf)
        .groupby(["lr", "batch_size", "model", "optimizer"])[f"train/PCC"]
        .mean()
        .replace(np.inf, np.nan)
        .dropna()
        .reset_index()
    )
    # Find the best lr, optimizer combination for each batch_size based on train/PCC
    best_lr_per_batch = best_df.groupby(["batch_size", "model"])["train/PCC"].idxmax()
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
    # as grouped bar chart, x-axis is batch size, y-axis is train/PCC, grouped by model
    g = sns.catplot(
        data=summary_df,
        x="batch size",
        y="train/PCC",
        hue="model",
        hue_order=ORDER,
        kind="bar",
        palette=colorblind_palette,
        height=HEIGHT,
        aspect=WIDTH/HEIGHT,
    )
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    #put legend below plot
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
    plt.tight_layout()
    g.set(ylim=(0, 1))
    plt.savefig(
        f"plots/StochasticCCA/{data}_models_different_batch_sizes.svg", bbox_inches="tight")
    plt.close()
    # Also return the average train/PCC for each model at each batch size as a latex table
    summary_df.groupby(["model", "batch size"])["train/PCC"].mean().unstack().to_latex(
        f"plots/StochasticCCA/{data}_models_different_batch_sizes.tex"
    )


if __name__ == "__main__":
    # plot_all_learning_curves(data="mediamill", batch_sizes=[5, 100])
    # plot_all_learning_curves(data="cifar", batch_sizes=[5, 100])
    plot_models_different_batch_sizes("mediamill")
    plot_models_different_batch_sizes("cifar")
