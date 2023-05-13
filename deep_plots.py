import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from wandb_utils import get_summary, get_run_data

PROJECT = "DeepDeltaEigenGame"

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
DASHES = [(0, 0), (2, 2), (2, 2), (2, 2)]
MODEL_TO_TITLE = {
    "DCCAEY": "DCCA-EY",
    "DCCAGH": "DCCA-GH",
    "DCCANOI": "DCCA-NOI",
    "DCCA": "DCCA-STOL-100",
}

ORDER = ["DCCA-EY", "DCCA-GH", "DCCA-NOI", "DCCA-STOL-100"]


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
    # map model names to titles
    run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
    #rename val/corr to Validation TCC
    run_data = run_data.rename(columns={"val/corr": "Validation TCC"})
    plt.figure()
    sns.lineplot(
        data=run_data,
        x="epoch",
        y="Validation TCC",
        hue=hue,
        hue_order=ORDER,
    )
    plt.title(rf"Top 50 DCCA on {data}")
    plt.savefig(f"plots/{title}.png")


def plot_ey_lr():
    id_df, summary_df, config_df = get_summary(project=PROJECT)
    summary_df = pd.concat([id_df, summary_df, config_df], axis=1)
    summary_df = summary_df.loc[summary_df["data"] == 'SplitMNIST']
    #For DCCAEY get all runs
    summary_df = summary_df.loc[summary_df["model"] == "DCCAEY"]
    run_data = get_run_data(ids=summary_df["id"].tolist(), project=PROJECT)
    # map model names to titles
    run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
    # rename val/corr to Validation TCC
    run_data = run_data.rename(columns={"val/corr": "Validation TCC"})
    plt.figure()
    sns.lineplot(
        data=run_data,
        x="epoch",
        y="Validation TCC",
        hue="lr",
        hue_order=ORDER,
    )


    plt.title('Top 50 DCCA on Split MNIST For DCCA-EY With\n Different Learning Rates', wrap=True)
    plt.savefig(f"plots/dcca_lr_experiment.png")


#plot_ey_lr()
plot_all_models(data="XRMB")

# for data in ["SplitMNIST", "XRMB"]:
#     for batch_size in [100]:
#         plot_tcc(data=data)
