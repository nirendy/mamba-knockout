from functools import lru_cache
from typing import Callable, TypeVar, assert_never, cast

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from datasets import (
    load_dataset as huggingface_load_dataset,
)

from src.core.consts import COUNTER_FACT_2_KNOWN1000_COL_CONV, DATASETS_IDS, PATHS
from src.core.names import COLS, DatasetName
from src.core.types import ALL_SPLITS_LITERAL, SPLIT, TPromptData, TPromptDataFlat, TPromptOriginalIndex, TSplitChoise
from src.data_ingestion.datasets.splitting import split_dataset


def load_splitted_counter_fact(
    split: TSplitChoise = (SPLIT.TRAIN1,),
    add_split_name_column: bool = False,
    align_to_known: bool = False,
) -> Dataset:
    splitted_path = PATHS.dataset_dir(DatasetName.counter_fact) / "splitted"

    if not splitted_path.exists():
        print("Creating splitted dataset")
        dataset_name = DATASETS_IDS[DatasetName.counter_fact]
        num_splits = 5
        split_ratio = 0.1
        seed = 42

        dataset = huggingface_load_dataset(dataset_name)["train"]  # type: ignore

        splitted_dataset = split_dataset(dataset, num_splits, split_ratio, seed)
        splitted_dataset.save_to_disk(str(splitted_path))

    data: DatasetDict = load_from_disk(str(splitted_path))  # type: ignore

    if split == ALL_SPLITS_LITERAL:
        split = list(data.keys())
    if isinstance(split, str):
        split = [SPLIT(split)]

    datasets = [data[split_name] for split_name in split]

    if add_split_name_column:
        for i, (split_name, dataset) in enumerate(zip(split, datasets)):
            dataset = dataset.add_column("split_name", [split_name] * len(dataset))  # type: ignore
            datasets[i] = dataset

    dataset = concatenate_datasets(datasets)

    if align_to_known:
        for (
            counter_fact_col,
            known1000_col,
        ) in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
            dataset = dataset.rename_column(counter_fact_col, known1000_col)
        dataset = dataset.remove_columns([COLS.COUNTER_FACT.TARGET_FALSE, COLS.COUNTER_FACT.TARGET_FALSE_ID])
    return dataset


@lru_cache(maxsize=1)
def get_prompt_ids(dataset_name: DatasetName, split: TSplitChoise = ALL_SPLITS_LITERAL) -> list[TPromptOriginalIndex]:
    match dataset_name:
        case DatasetName.counter_fact:
            dataset = load_splitted_counter_fact(
                split,
            )
            return dataset[COLS.ORIGINAL_IDX]
    assert_never(dataset_name)


def get_raw_data(dataset_name: DatasetName) -> TPromptDataFlat:
    match dataset_name:
        case DatasetName.counter_fact:
            dataset = load_splitted_counter_fact(
                ALL_SPLITS_LITERAL,
            )
            return TPromptDataFlat(pd.DataFrame(cast(dict, dataset)))
    assert_never(dataset_name)


def get_indexed_raw_data(dataset_name: DatasetName) -> TPromptData:
    df = get_raw_data(dataset_name)
    return flat_to_indexed_prompt_data(df)


def indexed_to_flat_prompt_data(df: TPromptData) -> TPromptDataFlat:
    return TPromptDataFlat(df.reset_index())


def flat_to_indexed_prompt_data(df: TPromptDataFlat) -> TPromptData:
    return TPromptData(df.set_index(COLS.ORIGINAL_IDX))


_T_DF = TypeVar("_T_DF", bound=pd.DataFrame)


def df_safe_operation(df: _T_DF, operation: Callable[[_T_DF], pd.DataFrame]) -> _T_DF:
    # Workaround to ensure type checking, it's on the user to ensure the operation is not changinging the index
    return cast(_T_DF, operation(df))
