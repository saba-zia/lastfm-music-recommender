from pathlib import Path

import pandas as pd

from .config import (
    INTERACTIONS_FILE,
    ITEMS_FILE,
    USERS_FILE,
    MUSICNN_FILE,
    LOCAL_TRAIN_FILE,
    LOCAL_VAL_FILE,
)


def validate_file_exists(file_path: Path) -> None:
    """
    Check whether a required dataset file exists.

    Parameters
    ----------
    file_path:
        Path to the dataset file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required data file was not found: {file_path}\n"
            "Place the dataset file inside data/raw/."
        )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Validate that a dataframe contains all required columns.

    Parameters
    ----------
    dataframe:
        Dataframe to validate.
    required_columns:
        Columns expected in the dataframe.
    dataset_name:
        Human-readable dataset name used in error messages.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def read_tsv(file_path: Path) -> pd.DataFrame:
    """
    Read a tab-separated dataset file.

    Parameters
    ----------
    file_path:
        Path to the TSV-like file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    validate_file_exists(file_path)

    return pd.read_csv(file_path, sep="\t")


def load_interactions() -> pd.DataFrame:
    """
    Load the original user-item interaction dataset.

    Returns
    -------
    pandas.DataFrame
        Interaction dataframe containing user_id, item_id and count.
    """
    dataframe = read_tsv(INTERACTIONS_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={"user_id", "item_id", "count"},
        dataset_name="Interaction dataset",
    )

    return dataframe


def load_items() -> pd.DataFrame:
    """
    Load item metadata.

    Returns
    -------
    pandas.DataFrame
        Item metadata containing artist, song, album and genre.
    """
    dataframe = read_tsv(ITEMS_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={
            "item_id",
            "artist",
            "song",
            "album_name",
            "genre",
        },
        dataset_name="Item metadata",
    )

    return dataframe


def load_users() -> pd.DataFrame:
    """
    Load user metadata.

    Returns
    -------
    pandas.DataFrame
        User metadata dataframe.
    """
    dataframe = read_tsv(USERS_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={
            "user_id",
            "country",
            "age_at_registration",
            "gender",
            "registration_date",
        },
        dataset_name="User metadata",
    )

    dataframe["registration_date"] = pd.to_datetime(
        dataframe["registration_date"],
        errors="coerce",
    )

    return dataframe


def load_musicnn_features() -> pd.DataFrame:
    """
    Load MusicNN audio feature vectors.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing item_id and MusicNN features.
    """
    dataframe = read_tsv(MUSICNN_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={"item_id"},
        dataset_name="MusicNN feature dataset",
    )

    return dataframe


def load_local_train() -> pd.DataFrame:
    """
    Load the local training split.

    Returns
    -------
    pandas.DataFrame
        Local training interactions.
    """
    dataframe = read_tsv(LOCAL_TRAIN_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={"user_id", "item_id", "events"},
        dataset_name="Local training dataset",
    )

    return dataframe


def load_local_validation() -> pd.DataFrame:
    """
    Load the local validation split.

    Returns
    -------
    pandas.DataFrame
        Local validation interactions.
    """
    dataframe = read_tsv(LOCAL_VAL_FILE)

    validate_columns(
        dataframe=dataframe,
        required_columns={"user_id", "item_id", "events"},
        dataset_name="Local validation dataset",
    )

    return dataframe