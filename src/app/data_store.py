import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from streamlit import cache_resource

from src.analysis.experiment_results.default_data_reqs import get_default_data_reqs
from src.analysis.experiment_results.helpers import (
    get_model_evaluations,
)
from src.analysis.experiment_results.model_prompt_combination import get_model_combinations_prompts
from src.analysis.experiment_results.results_bank import (
    RESULTS_BASE_PATH,
    get_experiment_results_bank,
)
from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
)
from src.core.names import COLS, DatasetName
from src.core.types import (
    MODEL_ARCH_AND_SIZE,
    TCodeVersionName,
    TPromptData,
    TPromptOriginalIndex,
)
from src.data_ingestion.data_defs.data_defs import (
    DataReqs,
    ModelCombinationsPrompts,
    PromptNew,
    Prompts,
    ResultBank,
    SummarizedDataFulfilledReqs,
    Tokenizers,
)
from src.data_ingestion.datasets.download_dataset import get_raw_data
from src.utils.streamlit.helpers.cache import CacheWithDependencies


@CacheWithDependencies()
def load_model_evaluations(code_version: TCodeVersionName, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> TPromptData:
    return get_model_evaluations(code_version, [model_arch_and_size])[model_arch_and_size]


@CacheWithDependencies()
def load_model_evaluations_dict(code_version: TCodeVersionName) -> dict[MODEL_ARCH_AND_SIZE, TPromptData]:
    """Load evaluation data for all models with caching"""
    return {
        model_arch_and_size: load_model_evaluations(code_version, model_arch_and_size)
        for model_arch_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
    }


@cache_resource
def merge_model_evaluations_streamlit_rendered(code_version: TCodeVersionName) -> StreamlitRenderer:
    """Load evaluation data for all models with caching"""
    return StreamlitRenderer(
        pd.concat(
            [
                df.assign(model_arch=key.arch, model_size=key.size)
                for key, df in load_model_evaluations_dict(code_version).items()
            ]
        ),
        spec=f"model_evals_{code_version}.csv",
        spec_io_mode="rw",
    )


@CacheWithDependencies(disable_cache=False)
def load_results_bank() -> ResultBank:
    return get_experiment_results_bank()


@CacheWithDependencies()
def load_test_results_bank() -> ResultBank:
    return get_experiment_results_bank(results_base_paths=(RESULTS_BASE_PATH.TEST,))


@CacheWithDependencies()
def get_tokenizerults_bank() -> ResultBank:
    return get_experiment_results_bank(results_base_paths=(RESULTS_BASE_PATH.TEST,))


@CacheWithDependencies()
def load_prompts(
    dataset: DatasetName = DatasetName.counter_fact,
) -> Prompts:
    df = get_raw_data(dataset)

    return Prompts(
        {TPromptOriginalIndex(int(row[COLS.ORIGINAL_IDX])): PromptNew(dict(row)) for _, row in df.iterrows()},
    )


# Data Requirements hooks
@CacheWithDependencies(disable_cache=False)
def load_fulfilled_reqs_df() -> SummarizedDataFulfilledReqs:
    """Load the data requirements options and overrides to dispaly the fulfilled requirements"""
    results_bank = load_results_bank()
    items = get_default_data_reqs().to_dict()
    # items = {k: v for i, (k, v) in enumerate(items.items()) if i < 5}
    options = DataReqs(items).to_fulfilled_reqs(results_bank)
    return SummarizedDataFulfilledReqs(options)


@CacheWithDependencies()
def get_merged_evaluations(prompt_idx: TPromptOriginalIndex, code_version: TCodeVersionName) -> pd.DataFrame:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - DataFrame with model-specific evaluations merged
    """
    model_evaluations = load_model_evaluations_dict(code_version)

    # Create list to hold each model's evaluation
    model_evals = []

    for model_combination in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_df = model_evaluations[model_combination]
        if prompt_idx not in model_df.index:
            continue

        row = model_df.loc[prompt_idx]

        # Filter out prompt-related columns (they're the same for all models)
        model_specific_data = {
            col: val for col, val in row.items() if col not in GLOBAL_APP_CONSTS.PROMPT_RELATED_COLUMNS
        }

        model_specific_data["model_arch"] = model_combination.arch
        model_specific_data["model_size"] = model_combination.size

        model_evals.append(model_specific_data)

    return pd.DataFrame(model_evals)


@CacheWithDependencies()
def load_model_combinations_prompts(
    code_version: TCodeVersionName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE], seed: int
) -> ModelCombinationsPrompts:
    return (
        ModelCombinationsPrompts(get_model_combinations_prompts(code_version, model_arch_and_sizes, seed))
        .sort_by_prompt_count()
        .change_chosen_prompt_by_seed(seed)
    )


@CacheWithDependencies(is_resource=True, max_entries=1)
def load_unique_tokenizers(
    model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE],
) -> Tokenizers:
    return Tokenizers.from_unique_tokenizers(model_arch_and_sizes)
