import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
DASHES = [(0, 0), (2, 2), (2, 2), (2, 2)]
MODEL_TO_TITLE = {'DCCAGEPGD': 'DCCA-GEPGD',
                  'DCCANOI': 'DCCA-NOI',
                  'DCCA': 'DCCA-STOL-100',
                  'DCCA-STOL-500': 'DCCA-STOL-500'}
X_AXIS="time"

def get_run_data(ids=None, mode='Train'):
    api = wandb.Api()
    runs = api.runs("jameschapman/DCCAGame")
    data = []
    for run in runs:
        if ids is None or run.id in ids:
            if mode == 'Train':
                history = pd.DataFrame(run.scan_history(keys=["epoch", "train/corr", "_runtime"]))
            else:
                history = pd.DataFrame(run.scan_history(keys=["epoch", "val/corr", "_runtime"]))
            history["model"] = run.config["model"]
            data.append(history)
    if len(data) > 0:
        return pd.concat(data).reset_index(drop=True)
    else:
        return None


def get_summary():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("jameschapman/DCCAGame")
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
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    import pandas as pd
    id_df = pd.DataFrame({'id': id_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df, summary_df, id_df], axis=1)
    all_df.to_csv("data/dcca.csv", index=False)


def filter_results(raw_results, batch_size, data='mnist', mode='Train'):
    if mode == 'Train':
        y = "train/corr"
    else:
        y = "val/corr"
    raw_results = raw_results[raw_results["data"] == data]
    results = raw_results[raw_results["batch_size"] == batch_size]
    results["model"] = results["model"].map(MODEL_TO_TITLE)
    results = results[results["model"].isin(MODEL_TO_TITLE.values())]
    dcca_500_results = raw_results[(raw_results["model"] == "DCCA") & (raw_results["batch_size"] == 500)]
    dcca_500_results["model"] = "DCCA-STOL-500"
    dcca_100_results = raw_results[(raw_results["model"] == "DCCA") & (raw_results["batch_size"] == 100)]
    dcca_100_results["model"] = "DCCA-STOL-100"
    results = pd.concat([dcca_100_results, dcca_500_results, results]).drop_duplicates()
    results = results[results["epoch"] == 29]
    results = results.sort_values(by=y, ascending=False)
    return results.groupby('model').head(1).reset_index(drop=True)


def plot_cc(results, batch_size, data='mnist', mode='Train', noisymnist=False):
    os.makedirs(f"DCCA/{data}/{batch_size}", exist_ok=True)
    if mode == 'Train':
        y = "train/corr"
    else:
        y = "val/corr"
    for x,x_label, file_label in zip(["_runtime", "epoch"], ["Time (s)", "Epoch"],["time", "epoch"]):
        plt.figure()
        sns.lineplot(
            data=results,
            x=x, y=y, hue="model", hue_order=MODEL_TO_TITLE.values(), style="model",
            style_order=MODEL_TO_TITLE.values()
        ).set_title(f"{mode} Correlation for {data} with batch size {batch_size}")
        plt.xlabel(x_label)
        plt.ylabel("Correlation Captured")
        if noisymnist:
            plt.ylim(0, 15)
        else:
            plt.ylim(0, 50)
        plt.savefig(f"DCCA/{data}/{batch_size}/pcc_{mode}_{file_label}.png")


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    get_summary()
    results = pd.read_csv("data/dcca.csv")
    for batch_size in [100]:
        for data in ['XRMB', 'SplitMNIST', 'NoisyMNIST']:
            for mode in ['Validation']:
                data_results = filter_results(results, batch_size, data, mode=mode)
                run_data = get_run_data(data_results[data_results["model"] != 'DCCA-STOL-500']["id"].tolist(),
                                        mode=mode)
                run_data["model"] = run_data["model"].map(MODEL_TO_TITLE)
                stol_500_run_data = get_run_data(
                    data_results[data_results["model"] == 'DCCA-STOL-500']["id"].tolist(), mode=mode)
                if stol_500_run_data is not None:
                    stol_500_run_data["model"] = "DCCA-STOL-500"
                    run_data = pd.concat([run_data, stol_500_run_data])
                plot_cc(run_data.reset_index(), batch_size, data=data, mode=mode, noisymnist=data == 'NoisyMNIST')
