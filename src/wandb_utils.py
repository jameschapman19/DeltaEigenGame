import pandas as pd
import wandb


def get_run_data(ids=None, project=""):
    api = wandb.Api(timeout=20)
    runs = api.runs(f"{project}")
    data = []
    for run in runs:
        if ids is None or run.id in ids:
            history = pd.DataFrame(run.history())
            history["model"] = run.config["model"]
            history["lr"] = run.config["lr"]
            history["batch_size"] = run.config["batch_size"]
            data.append(history)
    data = pd.concat(data).reset_index(drop=True)
    return data


def get_summary(project=""):
    api = wandb.Api(timeout=60)

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{project}")
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
