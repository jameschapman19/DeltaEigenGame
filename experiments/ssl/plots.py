"""
Generates plots of results from wandb api
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.wandb_utils import get_summary
import wandb
# Set a consistent color scheme for NeurIPS paper
palette = "colorblind"
colorblind_palette = sns.color_palette(palette, as_cmap=True)
sns.set_style("whitegrid")
sns.set_context(
    "paper", font_scale=2.0, rc={"lines.linewidth": 2.5, "axes.labelsize": 16}
)
# Set the default figure size
plt.rcParams["figure.figsize"] = (18, 6)  # Adjust the values as needed
os.makedirs("plots/SSL", exist_ok=True)

CIFAR10_names = ["eygep-cifar10", "barlow_twins-cifar10", "vicreg-cifar10"]
CIFAR100_names = ["eygep-cifar100","barlow_twins-cifar100", "vicreg-cifar100"]

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

def plot_learning_curve(cifar=100):
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
    plot = sns.lineplot(
        data=df,
        x="epoch",
        y="val_acc1",
        hue="name",
        hue_order=ORDER,
        palette=colorblind_palette,
    )
    plot.set_title(f"CIFAR-{cifar} Top-1 Accuracy vs. Epoch")
    plot.set_xlabel("Epoch")
    plot.set_ylabel(f"CIFAR-{cifar} Top-1 Accuracy")
    plot.legend(loc="lower right")
    plt.savefig(f"plots/SSL/cifar{cifar}_top1_learning_curve.svg")
    plt.close()
    plot = sns.lineplot(
        data=df,
        x="epoch",
        y="val_acc5",
        hue="name",
        hue_order=ORDER,
        palette=colorblind_palette,
    )
    plot.set_title(f"CIFAR-{cifar} Top-5 Accuracy vs. Epoch")
    plot.set_xlabel("Epoch")
    plot.set_ylabel(f"CIFAR-{cifar} Top-5 Accuracy")
    plot.legend(loc="lower right")
    plt.savefig(f"plots/SSL/cifar{cifar}_top5_learning_curve.svg")
    plt.close()

def plot_projector_ablation():
    """
    Plot projector dimension vs. performance for SSL-EY and SSL-SVD
    """
    # load data
    df = pd.read_excel("projector_dim.xlsx", sheet_name="Sheet1")

    for cifar in [10, 100]:
        plt.figure(figsize=(12, 6))
        g=sns.barplot(
            x="Projector Size",
            y=f"CIFAR-{cifar} Top-1",
            hue="Model",
            data=df,
        )
        plt.legend(loc="lower right")
        plt.title(f"CIFAR-{cifar} Top-1 Accuracy vs. Projector Dimension")
        plt.xlabel("Projector Dimension")
        plt.ylabel(f"CIFAR-{cifar} Top-1 Accuracy")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, title=None)
        plt.tight_layout()
        plt.savefig(f"plots/SSL/cifar{cifar}_proj_dim.svg", bbox_inches="tight")
        plt.close()


def main():
    plot_projector_ablation()
    plot_learning_curve(cifar=10)
    plot_learning_curve(cifar=100)


if __name__ == "__main__":
    main()
