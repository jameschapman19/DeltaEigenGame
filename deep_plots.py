import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np
from plots import get_run_data, get_summary

PROJECT = "DeepDeltaEigenGame"

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
DASHES = [(0, 0), (2, 2), (2, 2), (2, 2)]
MODEL_TO_TITLE = {'DCCAEY': 'DCCA-EY',
                    'DCCAGH': 'DCCA-GH',
                  'DCCANOI': 'DCCA-NOI',
                  'DCCA': 'DCCA-STOL-100'}

ORDER = ['DCCA-EY', 'DCCA-GH', 'DCCA-NOI', 'DCCA-STOL-100']

def get_best_runs(
        data="mnist", batch_size=100, objective="PCC", mode="Train", momentum=0.9, lr=None
):
    id_df, summary_df, config_df = get_summary()
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    if lr is not None:
        summary_df = summary_df.loc[summary_df["lr"] == lr]
    best_df = summary_df.sort_values(by=[f"val/corr"], ascending=False)
    best_df = best_df.groupby("model").head(1).reset_index(drop=True)
    # get run data for models in summary_df matching best_df
    summary_df = pd.merge(
        best_df, summary_df, on=["rho","lr"], how="left"
    )
    df = get_run_data(ids=summary_df["id"].tolist())
    # Change column title _step to samples seen
    df = df.rename(columns={"_step": "Samples Seen"})
    return df


def plot_tcc(data="mnist", batch_size=100, momentum=0.9, lr=None):
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
    plt.title(fr"Top 50 DCCA on {data}")
    if lr is None:
        lr = "tuned"
    plt.savefig(f"plots/{data}_{batch_size}_tcc_lr_{lr}.png")

for data in ["SplitMNIST", "XRMB"]:
    for batch_size in [100]:
        #plot_pvc(data=data, batch_size=batch_size, momentum=0, lr=lr)
        plot_tcc(data=data, batch_size=batch_size, lr=0.0001)