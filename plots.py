"""
Generates plots of results from wandb api
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import numpy as np

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

PROJECT = "GEP-GD"

MODEL_TO_TITLE = {
    "eckhart": "GEP-GD",
    "eckhartordered": "k-GEP-GD",
    "delta": r"k-GEP-GD",
    "subspace": r"GEP-GD",
    "gha": r"GHA",
    "sgha": "SGHA",
    "gamma": r"$\gamma$" + "-EigenGame",
    "sp": "Stochastic Power",
}

# Set order of models in plots
ORDER = [
    "GEP-GD",
    #"Eckhart Ordered",
    #"k-GEP-GD",
    #"GEP-GD",
    #"GHA",
    "SGHA",
    r"$\gamma$" + "-EigenGame",
    "Stochastic Power",
]


DIMENSIONS = {
    "mnist": (392, 392),
    "cifar": (1536, 1536),
    "mediamill": (120, 120),
}

def get_run_data(ids=None):
    api = wandb.Api(timeout=20)
    runs = api.runs(f"jameschapman/{PROJECT}")
    data = []
    for run in runs:
        if ids is None or run.id in ids:
            history = pd.DataFrame(run.history())
            history["model"] = run.config["model"]
            data.append(history)
    return pd.concat(data).reset_index(drop=True)


def get_summary():
    api = wandb.Api(timeout=60)

    # Project is specified by <entity/project-name>
    runs = api.runs(f"jameschapman/{PROJECT}")
    summary_list = []
    config_list = []
    name_list = []
    id_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        id_list.append(run.id)
        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    import pandas as pd

    id_df = pd.DataFrame({"id": id_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)

    return id_df, summary_df, config_df


def get_best_runs(
        data="mnist", batch_size=100, objective="PCC", mode="Train", momentum=0.9, lr=None
):
    id_df, summary_df, config_df = get_summary()
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
    df = get_run_data(ids=summary_df["id"].tolist())
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
    plt.title(fr"Top 4 CCA on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})")
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
    plt.title(fr"Top 4 PLS on {data} ($d_x$={DIMENSIONS[data][0]}, $d_y$={DIMENSIONS[data][1]})")
    if lr is None:
        lr = "tuned"
    plt.savefig(f"plots/{data}_{batch_size}_pvc_lr_{lr}.png")


for data in ["mnist", "cifar", "mediamill"]:
    for batch_size in [100]:
        for lr in [0.0001,0.001, 0.01]:
            plot_pvc(data=data, batch_size=batch_size, momentum=0, lr=lr)
            plot_pcc(data=data, batch_size=batch_size, momentum=0, lr=lr)

