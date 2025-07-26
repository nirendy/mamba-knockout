"""
EvaluateModelExperiment: Experiment for evaluating model performance on datasets

In this experiment implementation:
The sub-task is a batch of prompts
The inner loop is running the model on each prompt and collecting detailed metrics
The sub task result is a DataFrame with model predictions and metrics
The combined result is saved as a CSV file
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, cast

import pandas as pd
import torch
from cachetools import LRUCache, cached
from tqdm import tqdm

from src.core.consts import COUNTER_FACT_2_KNOWN1000_COL_CONV
from src.core.names import COLS, ExperimentName
from src.core.types import MODEL_ARCH, TPromptData, TPromptDataFlat
from src.data_ingestion.datasets.download_dataset import (
    flat_to_indexed_prompt_data,
    get_raw_data,
)
from src.data_ingestion.helpers.logits_utils import (
    generate_next_tokens,
    get_subj_idx,
    logits_to_probs,
    trim_left_and_right_pad,
)
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams


@dataclass(frozen=True)
class EvaluateModelParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.evaluate_model)
    drop_subject: bool = False
    drop_subj_last_token: bool = False
    with_3_dots: bool = False
    new_max_tokens: int = 5
    top_k_tokens: int = 5


@lru_cache(maxsize=100)
def _get_output_path(evaluate_runner: EvaluateModelRunner) -> TPromptDataFlat:
    df = pd.read_csv(evaluate_runner.output_result_path, index_col=False)
    for (
        counter_fact_col,
        known1000_col,
    ) in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
        if counter_fact_col not in df.columns:
            assert known1000_col in df.columns
            df[counter_fact_col] = df[known1000_col]

        if known1000_col in df.columns:
            df = df.drop(columns=[known1000_col])

    return TPromptDataFlat(df)


evalurate_model_prompt_data_cache = LRUCache(maxsize=30)


@dataclass(frozen=True)
class EvaluateModelRunner(BaseRunner[EvaluateModelParams]):
    """Configuration for model evaluation."""

    variant_params: EvaluateModelParams

    @staticmethod
    def _get_variant_params():
        return EvaluateModelParams

    @classmethod
    def get_variant_output_keys(cls):
        return super().get_variant_output_keys()

    @property
    def output_result_path(self) -> Path:
        return self.variation_paths.outputs_path / "outputs.csv"

    def get_outputs(self) -> TPromptDataFlat:  # type: ignore
        return _get_output_path(self)

    @cached(evalurate_model_prompt_data_cache)
    def get_prompt_data(self) -> TPromptData:
        df = flat_to_indexed_prompt_data(self.get_outputs())
        from src.analysis.prompt_filterations import AnyExistingCompletePromptFilteration

        if not isinstance(self.input_params.filteration, AnyExistingCompletePromptFilteration):
            df = df.loc[self.input_params.filteration.contextualize(self).get_prompt_ids()]
        return cast(
            TPromptData,
            df,
        )

    def _compute_impl(self) -> None:
        run(self)

    @lru_cache(maxsize=1)
    def is_computed(self) -> bool:  # type: ignore
        return self.output_result_path.exists()

    def get_runner_dependencies(self):
        return self.input_params.filteration.contextualize(self).get_dependencies()


def run(args: EvaluateModelRunner):
    print(args)
    if args.output_result_path.exists() and not args.metadata_params.overwrite_existing_outputs:
        print(f"Output file {args.output_result_path} already exists")
        return

    args.create_experiment_dir()
    df = get_raw_data(args.input_params.dataset_name)

    model_interface = args.variant_params.get_model_interface()

    model = model_interface.model
    tokenizer = model_interface.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = model_interface.device

    acc = 0

    df[COLS.EVALUATE_MODEL.MODEL_CORRECT] = False
    df[COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE] = 0.0
    df[COLS.EVALUATE_MODEL.TARGET_RANK] = None
    df[COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS] = None
    df[COLS.EVALUATE_MODEL.MODEL_GENERATION] = None
    df[COLS.EVALUATE_MODEL.TARGET_PROBS] = 0.0
    df[COLS.EVALUATE_MODEL.TARGET_TOKENS] = None
    for counter_fact_col, known1000_col in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
        if known1000_col in df.columns:
            df[counter_fact_col] = df[known1000_col]

    pbar = tqdm(range(0, len(df), args.effective_batch_size), total=len(df) // args.effective_batch_size)
    for start_idx in pbar:
        idx = df.index[start_idx : start_idx + args.effective_batch_size]
        input_prompt = df.loc[idx, COLS.COUNTER_FACT.PROMPT]
        target = df.loc[idx, COLS.COUNTER_FACT.TARGET_TRUE]

        target_token_idx_padded = tokenizer(
            target.to_list(),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",  # type: ignore
        )["input_ids"]

        if args.variant_params.model_arch == MODEL_ARCH.LLAMA2:
            target_token_idx_padded = trim_left_and_right_pad(
                target_token_idx_padded,
                trim_value=29871,
                pad_value=tokenizer.pad_token_id,  # type: ignore
            )

        target_first_token_idx = target_token_idx_padded[:, 0].unsqueeze(1)  # type: ignore
        df.loc[idx, COLS.EVALUATE_MODEL.TARGET_TOKENS] = list(
            map(
                json.dumps,
                map(
                    lambda x: tokenizer.batch_decode(x, skip_special_tokens=True),
                    [
                        lst[
                            : next(
                                (i for i in range(len(lst) - 1, -1, -1) if lst[i] != tokenizer.pad_token_id),
                                -1,
                            )
                            + 1
                        ]
                        for lst in target_token_idx_padded.tolist()  # type: ignore
                    ],
                ),
            )
        )

        if args.variant_params.with_3_dots:
            input_prompt += " ..."
        if args.variant_params.drop_subject:
            input_prompt = input_prompt.replace(df.loc[idx, COLS.COUNTER_FACT.SUBJECT], "")
        elif args.variant_params.drop_subj_last_token:
            subj_idx = get_subj_idx(input_prompt, df.loc[idx, COLS.COUNTER_FACT.SUBJECT], tokenizer)  # type: ignore

        input_ids = tokenizer(input_prompt.to_list(), return_tensors="pt", padding=True)["input_ids"]

        if args.variant_params.drop_subj_last_token:
            input_ids = input_ids[:subj_idx] + input_ids[subj_idx + 1 :]  # type: ignore

        input_ids = input_ids.to(device)  # type: ignore

        # TODO: the logits of the next token is different for different the amount token generated, understand why
        _, first_logits, new_input_ids = generate_next_tokens(
            model, input_ids, args.variant_params.new_max_tokens, args.variant_params.model_arch
        )

        # Get the next token probs
        next_probs = logits_to_probs(first_logits)

        # Get the top k outputs and their probs
        top_probs, top_indices = map(torch.Tensor.tolist, torch.topk(next_probs, args.variant_params.top_k_tokens))
        top_tokens = list(map(tokenizer.batch_decode, top_indices))  # type: ignore
        top_outputs = list(
            map(
                list,
                map(
                    lambda x: zip(*x),
                    zip(
                        top_indices,
                        top_tokens,
                        top_probs,
                    ),
                ),
            )
        )

        target_probs = torch.gather(
            next_probs,
            1,
            target_first_token_idx.to(next_probs.device),
        )

        # Get the rank of the target token
        target_rank = (next_probs > target_probs).sum(dim=-1) + 1
        df.loc[idx, COLS.EVALUATE_MODEL.TARGET_RANK] = target_rank.tolist()
        df.loc[idx, COLS.EVALUATE_MODEL.TARGET_PROBS] = target_probs.squeeze(1).tolist()
        df.loc[idx, COLS.EVALUATE_MODEL.MODEL_CORRECT] = (target_rank == 1).tolist()
        df.loc[idx, COLS.EVALUATE_MODEL.MODEL_OUTPUT] = list(map(lambda x: x[0], top_tokens))
        df.loc[idx, COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE] = list(map(lambda x: x[0], top_probs))
        df.loc[idx, COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS] = list(map(json.dumps, top_outputs))
        df.loc[idx, COLS.EVALUATE_MODEL.MODEL_GENERATION] = tokenizer.batch_decode(new_input_ids)

        acc += df.loc[idx, COLS.EVALUATE_MODEL.MODEL_CORRECT].sum()

    print(acc / len(df))
    df.to_csv(args.output_result_path, index=False)
