"""
Generates plots of results from wandb api
"""
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
# Set the default figure size
plt.rcParams["figure.figsize"] = (18, 6)  # Adjust the values as needed
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

os.makedirs("plots/StochasticCCA", exist_ok=True)


def get_best_runs(
    data="mnist",
    batch_size=100,
    objective="PCC",
    mode="Train",
    lr=None,
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
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


def plot_pcc(data="mnist", batch_size=100, lr=None):
    # Plot PCC for best runs for each model
    df = get_best_runs(
        data=data,
        batch_size=batch_size,
        objective="PCC",
        mode="Train",
        lr=lr,
    )
    # # drop nans in "train/PCC" column
    # df = df.dropna(subset=["train/PCC"])
    # map model names to titles
    df["model"] = df["model"].map(MODEL_TO_TITLE)
    df = df.rename(columns={"batch_size": "batch size"})
    df = df.rename(columns={"train/PCC": "Train PCC"})
    df = df.rename(columns={"samples_seen": "Samples Seen"})
    # Fill NaN values in "Samples Seen" column with previous values
    df["Samples Seen"].fillna(method="ffill", inplace=True)
    # drop rows with Nan in "Train PCC" column
    df = df.dropna(subset=["Train PCC"])
    plt.figure(figsize=(11.87, 8.27))
    sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="model",
        hue_order=ORDER,
        palette=sns.color_palette(palette, n_colors=3),
    )
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    if lr is None:
        lr = "tuned"
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"plots/StochasticCCA/{data}_{batch_size}_pcc_lr_{lr}.svg")


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
    summary_df = pd.merge(best_lr_df, summary_df, on=["lr", "batch_size"], how="left")
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
    # Apply logarithmic transformation to 'lr'
    df["log_batch_size"] = np.log10(df["batch size"])
    plt.figure(figsize=(10, 5))
    sns_plot = sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="log_batch_size",
        # hue color palette
        palette=colorblind_palette,
    )
    # Get the handles and labels of the current plot's legend
    handles, labels = sns_plot.get_legend_handles_labels()

    # Map the transformed 'log_lr' back to original 'lr' for legend labels
    new_labels = [
        int(10 ** float(label)) for label in labels
    ]  # Skip the first label which is 'log_lr'

    # Update the legend
    sns_plot.legend(
        handles, new_labels, title="lr"
    )  # Skip the first handle which is 'log_lr'
    plt.ylim(0, 1)
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    plt.savefig(f"plots/StochasticCCA/{data}_minibatch_size_ablation.svg")


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
    plt.figure(figsize=(10, 5))
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


def plot_learning_rate_ablation(
    data="mnist",
    batch_size=100,
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    summary_df = summary_df.loc[summary_df["batch_size"] == batch_size]
    summary_df = summary_df.loc[summary_df["model"] == "ey"]
    summary_df = summary_df.loc[summary_df["optimizer"] == "Adam"]
    # Get the best run with Adam
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
    # Apply logarithmic transformation to 'lr'
    df["log_lr"] = np.log10(df["lr"])
    plt.figure(figsize=(10, 5))
    sns_plot = sns.lineplot(
        data=df,
        x="Samples Seen",
        y="Train PCC",
        hue="log_lr",
        # hue color palette
        palette=colorblind_palette,
    )
    # Get the handles and labels of the current plot's legend
    handles, labels = sns_plot.get_legend_handles_labels()

    # Map the transformed 'log_lr' back to original 'lr' for legend labels
    new_labels = [
        10 ** float(label) for label in labels
    ]  # Skip the first label which is 'log_lr'

    # Update the legend
    sns_plot.legend(
        handles, new_labels, title="lr"
    )  # Skip the first handle which is 'log_lr'
    plt.ylim(0, 1)
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    plt.savefig(f"plots/StochasticCCA/{data}_learning_rate_ablation.svg")


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
    )
    g.set(ylim=(0, 1))
    g.fig.set_figwidth(11.87)
    g.fig.set_figheight(8.27)
    plt.title(
        rf"Top 5 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})"
    )
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
    plt.tight_layout()
    plt.savefig(
        f"plots/StochasticCCA/{data}_models_different_batch_sizes.svg",
        bbox_inches="tight",
    )
    # Also return the average train/PCC for each model at each batch size as a latex table
    summary_df.groupby(["model", "batch size"])["train/PCC"].mean().unstack().to_latex(
        f"plots/StochasticCCA/{data}_models_different_batch_sizes.tex"
    )


if __name__ == "__main__":
    # plot_models_different_batch_sizes("mediamill")
    # plot_models_different_batch_sizes("cifar")
    for dataset in ["mediamill", "cifar"]:
        for batch_size in [50, 100]:
            plot_pcc(dataset, batch_size)
