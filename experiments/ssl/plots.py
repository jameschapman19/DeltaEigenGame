"""
Generates plots of results from wandb api
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from src.wandb_utils import get_summary
import wandb

# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set_style("whitegrid")
sns.set_context(
    "paper",
    font_scale=2.0,
    rc={"lines.linewidth": 2.5},
)
os.makedirs("plots/SSL", exist_ok=True)

CIFAR10_names = ["eygep-cifar10", "barlow_twins-cifar10", "vicreg-cifar10"]
CIFAR100_names = ["eygep-cifar100", "barlow_twins-cifar100", "vicreg-cifar100"]

NAMES_TO_TITLES = {
    "eygep-cifar10": "SSL-EY",
    "barlow_twins-cifar10": "Barlow Twins",
    "vicreg-cifar10": "VICReg",
    "eygep-cifar100": "SSL-EY",
    "barlow_twins-cifar100": "Barlow Twins",
    "vicreg-cifar100": "VICReg",
}

ORDER = ["SSL-EY", "Barlow Twins", "VICReg"]


def get_run_data(ids, project="solo-learn"):
    """
    Fetch run data from a Weights and Biases (wandb) project.

    Parameters:
    - ids (List[str] or None): List of specific run IDs to fetch; fetches all runs if None.
    - project (str): The name of the wandb project.

    Returns:
    - pd.DataFrame: Concatenated DataFrame of history data from runs.
    """
    api = wandb.Api(timeout=20)
    runs = api.runs(f"{project}")
    data = []

    for run in runs:
        if ids is None or run.id in ids:
            history = pd.DataFrame(run.history())
            history["name"] = run.config["name"]
            data.append(history)
    data = pd.concat(data).reset_index(drop=True)
    return data


def plot_learning_curve(cifar=100, plot_log_error=False):
    id_df, summary_df, config_df = get_summary(project="solo-learn")
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data/dataset"] == f"cifar{cifar}"]
    if cifar == 10:
        names = CIFAR10_names
    else:
        names = CIFAR100_names
    summary_df = summary_df.loc[summary_df["name"].isin(names)]
    # Get the run data
    df = get_run_data(ids=summary_df["id"].tolist(), project="solo-learn")
    # name to title
    df["name"] = df["name"].map(NAMES_TO_TITLES)

    # Rename val_acc1 to Top-1 Accuracy
    df = df.rename(columns={"val_acc1": "Top-1 Accuracy"})
    # Rename val_acc5 to Top-5 Accuracy
    df = df.rename(columns={"val_acc5": "Top-5 Accuracy"})

    if plot_log_error:
        df["Top-1 Accuracy"] = np.log((100 - df["Top-1 Accuracy"]) / 100)
        df["Top-5 Accuracy"] = np.log((100 - df["Top-5 Accuracy"]) / 100)
        ylabel = "Log (1 - Accuracy)"
    else:
        ylabel = "Accuracy"

    # put into long format so style= top-1 or top-5 and we plot accuracy
    df = pd.melt(
        df,
        id_vars=["name", "epoch"],
        value_vars=["Top-1 Accuracy", "Top-5 Accuracy"],
        var_name="style",
        value_name=ylabel,
    )
    # Initialize the figure with custom dimensions
    fig, ax = plt.subplots(figsize=(8, 4))  # 1.5 times wider than default (8, 8)
    plot = sns.lineplot(
        data=df,
        x="epoch",
        y=ylabel,
        hue="name",
        hue_order=ORDER,
        palette=colorblind_palette,
        style="style",
        ax = ax
    )
    plt.tight_layout(rect=[0,0,0.75,1])  # Make space for the legend
    # Update legend
    plt.legend(
        fontsize="x-small",
        bbox_to_anchor=(1.05, 0.5),  # Center align
        loc='center left',
        frameon=False  # Remove box
    )
    plot.set_title(f"CIFAR-{cifar} {ylabel} vs. Epoch")
    plot.set_xlabel("Epoch")
    plot.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"plots/SSL/cifar{cifar}_learning_curve{'_log_error' if plot_log_error else ''}.pdf", bbox_inches="tight")
    plt.close()


def plot_projector_ablation(plot_log_error=True):
    """
    Plot projector dimension vs. performance for SSL-EY and SSL-SVD
    """
    # load data
    df = pd.read_excel("projector_dim.xlsx", sheet_name="Sheet1")
    # drop model=SSL-SVD
    df = df.loc[df["Model"] != "SSL-SVD"]
    for cifar in [10, 100]:
        if plot_log_error:
            df[f"CIFAR-{cifar} Top-1"] = np.log((100 - df[f"CIFAR-{cifar} Top-1"])/100)
        g = sns.barplot(
            x="Projector Size",
            y=f"CIFAR-{cifar} Top-1",
            hue="Model",
            data=df,
        )
        plt.legend(loc="lower right")
        plt.title(f"CIFAR-{cifar} Log Error vs. Projector Dimension" if plot_log_error else f"CIFAR-{cifar} Top-1 Accuracy vs. Projector Dimension")
        plt.xlabel("Projector Dimension")
        ylabel = f"Log (1 - Top-1 Accuracy)" if plot_log_error else f"Top-1 Accuracy"
        plt.ylabel(ylabel)
        sns.move_legend(
            g,
            "upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            title=None,
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(f"plots/SSL/cifar{cifar}_proj_dim_log_error.pdf" if plot_log_error else f"plots/SSL/cifar{cifar}_proj_dim.pdf", bbox_inches="tight")
        plt.close()


def plot_corr_vs_acc(cifar=100, plot_log_error=False):
    id_df, summary_df, config_df = get_summary(project="solo-learn")
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data/dataset"] == f"cifar{cifar}"]
    if cifar == 10:
        names = ["eygep-cifar10-64"]
    else:
        names = ["eygep-cifar100-64"]
    summary_df = summary_df.loc[summary_df["name"].isin(names)]
    config_df = config_df.loc[config_df["name"].isin(names)]
    # Get the run data
    df = get_run_data(ids=summary_df["id"].tolist(), project="solo-learn")
    # name to title
    df["name"] = df["name"].map(NAMES_TO_TITLES)
    projector_dims = config_df["method_kwargs/proj_output_dim"].iloc[0]
    
    # Rename val_acc1 to Top-1 Accuracy
    df = df.rename(columns={"val_acc1": "Top-1 Validation Accuracy"})
    # Rename val_acc5 to Top-5 Accuracy
    df = df.rename(columns={"val_acc5": "Top-5 Validation Accuracy"})
    # Rename train_EYGEP_loss_epoch to EY
    df = df.rename(columns={"train_EYGEP_loss_epoch": "Train EY"})
    df["Train EY"] = df["Train EY"]*projector_dims

    if plot_log_error:
        df["Top-1 Validation Accuracy"] = np.log((100 - df["Top-1 Validation Accuracy"]) / 100)
        df["Top-5 Validation Accuracy"] = np.log((100 - df["Top-5 Validation Accuracy"]) / 100)
        ylabel = "Log (1 - Validation Accuracy)"
    else:
        ylabel = "Validation Accuracy"

    # put into long format so style= top-1 or top-5 and we plot accuracy
    df = pd.melt(
        df,
        id_vars=["name", "epoch","Train EY"],
        value_vars=["Top-1 Validation Accuracy"],#, "Top-5 Train Accuracy"],
        var_name="style",
        value_name=ylabel,
    )

    fig, ax1 = plt.subplots()

    # Plotting the accuracy losses
    sns.lineplot(
        data=df,
        x="epoch",
        y=ylabel,
        style="style",
        ax=ax1,
        legend='brief'
    )
    plt.legend(fontsize="x-small")

    # Create second y-axis for EY plot
    ax2 = ax1.twinx()

    # Plotting the EY loss
    sns.lineplot(
        data=df,
        x="epoch",
        y="Train EY",
        ax=ax2,
        color=sns.color_palette("tab10")[3],
        legend=False
    )

    # Customize plot
    ax1.set_title(f"CIFAR-{cifar}: {ylabel}" + r"& $\mathcal{L}_{EY}$ vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(ylabel, color=sns.color_palette("tab10")[0])
    ax2.set_ylabel(r"Train $\mathcal{L}_{EY}$", color=sns.color_palette("tab10")[3])

    # Coloring axes to match lines
    ax1.tick_params(axis='y', colors=sns.color_palette("tab10")[0])
    ax2.tick_params(axis='y', colors=sns.color_palette("tab10")[3])

    plt.tight_layout()

    # Save the plot
    filename = f"plots/SSL/cifar{cifar}_corr_vs_acc{'_log_error' if plot_log_error else ''}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def main():
    plot_corr_vs_acc(cifar=10, plot_log_error=True)
    plot_corr_vs_acc(cifar=100, plot_log_error=True)
    # plot_projector_ablation(plot_log_error=True)
    # plot_learning_curve(cifar=10, plot_log_error=True)
    # plot_learning_curve(cifar=100, plot_log_error=True)


if __name__ == "__main__":
    main()
