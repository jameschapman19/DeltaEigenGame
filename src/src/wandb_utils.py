import pandas as pd
from typing import List, Union, Tuple
import wandb


def get_run_data(ids: Union[None, List[str]] = None, project: str = "") -> pd.DataFrame:
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
            history["model"] = run.config["model"]
            history["lr"] = run.config["lr"]
            history["batch_size"] = run.config["batch_size"]
            history["optimizer"] = run.config["optimizer"]
            data.append(history)

    data = pd.concat(data).reset_index(drop=True)
    return data


def get_summary(project: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch summary data from a Weights and Biases (wandb) project.

    Parameters:
    - project (str): The name of the wandb project.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames containing IDs, summary metrics, and configurations.
    """
    api = wandb.Api(timeout=60)
    runs = api.runs(f"{project}")
    summary_list = []
    config_list = []
    id_list = []

    for run in runs:
        summary_list.append(run.summary._json_dict)
        id_list.append(run.id)
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

    id_df = pd.DataFrame({"id": id_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)

    return id_df, summary_df, config_df
