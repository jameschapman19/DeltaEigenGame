import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from wandb_utils import get_summary, get_run_data

PROJECT = "DeepDeltaEigenGame"

sns.set_context("paper", font_scale=2.0)
sns.set_style("whitegrid")
DASHES = [(0, 0), (2, 2), (2, 2), (2, 2)]
MODEL_TO_TITLE = {
    "DCCAEY": "DCCA-EY",
    "DCCAGH": "DCCA-GH",
    "DCCANOI": "DCCA-NOI",
    "DCCA": "DCCA-STOL-100",
    "DCCASimpler": "DCCA-SVD",
    "DCCAEY_NPSD": "DCCA-EY",
}

ORDER = ["DCCA-EY","DCCA-SVD", "DCCA-NOI", "DCCA-STOL-100"]


def get_best_runs(
    data="SplitMNIST", lr=None
):
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == data]
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
    best_df = summary_df.sort_values(by=[f"val/corr"], ascending=False)
    best_df = best_df.groupby("model").head(1).reset_index(drop=True)
    df = get_run_data(ids=best_df["id"].tolist(), project=PROJECT)
    return df

def plot_all_models(data="XRMB", lr=None):
    df = get_best_runs(data=data, lr=lr)
    plot_tcc(df, title=f"dcca_{data}")

def plot_tcc(run_data, title, data="XRMB", hue="model"):
    # drop DCCAEY
    run_data = run_data.loc[run_data["model"] != "DCCAEY"]
    # map model names to titles
    run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
    #rename val/corr to Validation TCC
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
    plt.savefig(f"plots/{title}.png")


def plot_simpler_lr():
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == 'SplitMNIST']
    # batch size 100
    summary_df = summary_df.loc[summary_df["batch_size"] == 100]
    #For DCCAEY get all runs
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
    plt.title('Top 50 DCCA on Split MNIST For DCCA-SVD With Different Learning Rates', wrap=True)
    plt.savefig(f"plots/dcca_lr_experiment.png")

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
    summary_df = pd.merge(
        best_df, summary_df, on=["lr", "batch_size"], how="left"
    )
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
    #tick every 5
    plt.xticks(np.arange(0, 21, 5))
    plt.title('Top 50 DCCA on Split MNIST For DCCA-SVD With\n Different Batch Sizes', wrap=True)
    plt.tight_layout()
    plt.savefig(f"plots/deep_{data}_minibatch_size_ablation.png")

plot_minibatch_size_ablation("SplitMNIST")
plot_simpler_lr()
plot_all_models(data="XRMB")

# for data in ["SplitMNIST", "XRMB"]:
#     for batch_size in [100]:
#         plot_tcc(data=data)
