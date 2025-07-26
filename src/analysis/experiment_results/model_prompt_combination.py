from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from src.analysis.experiment_results.helpers import get_model_evaluations
from src.core.names import COLS, ModelCombinationCols
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName, TPromptOriginalIndex


@dataclass
class ModelCombination:
    correct_models: set[MODEL_ARCH_AND_SIZE]
    incorrect_models: set[MODEL_ARCH_AND_SIZE]
    prompts: list[TPromptOriginalIndex]
    chosen_prompt: TPromptOriginalIndex

    def to_dict(self) -> dict[str, Any]:
        return {
            ModelCombinationCols.correct_models: list(self.correct_models),
            ModelCombinationCols.incorrect_models: list(self.incorrect_models),
            ModelCombinationCols.prompts: self.prompts,
            ModelCombinationCols.chosen_prompt: self.chosen_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCombination:
        return cls(
            correct_models={MODEL_ARCH_AND_SIZE(arch, size) for arch, size in data["correct_models"]},
            incorrect_models={MODEL_ARCH_AND_SIZE(arch, size) for arch, size in data["incorrect_models"]},
            prompts=data["prompts"],
            chosen_prompt=data["chosen_prompt"],
        )

    def choose_prompt_by_seed(self, seed: int) -> ModelCombination:
        random.seed(seed)
        chosen_prompt = random.choice(self.prompts)
        return ModelCombination(
            correct_models=self.correct_models,
            incorrect_models=self.incorrect_models,
            prompts=self.prompts,
            chosen_prompt=chosen_prompt,
        )


PROMPT_SELECTION_PATH = Path(__file__).parent / "prompt_selections.json"


def save_model_combinations_prompts(model_combinations: list[ModelCombination]) -> None:
    json.dump(
        [model_combination.to_dict() for model_combination in model_combinations],
        PROMPT_SELECTION_PATH.open("w"),
    )


def get_model_combinations_prompts(
    code_version: Optional[TCodeVersionName],
    model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE],
    seed: int,
) -> list[ModelCombination]:
    """Get all possible model combinations and their corresponding prompts.
    Each combination specifies which models should be correct and which should be incorrect.

    Returns:
        List of ModelCombination objects describing each valid combination pattern.
    """
    if PROMPT_SELECTION_PATH.exists():
        saved_selections = json.load(PROMPT_SELECTION_PATH.open("r"))
        saved_combinations = [ModelCombination.from_dict(selection) for selection in saved_selections]

        # Get all unique models from saved combinations
        saved_models = set()
        for combo in saved_combinations:
            saved_models.update(combo.correct_models)
            saved_models.update(combo.incorrect_models)

        # If requested models are a subset of saved models, derive from existing
        if saved_models - set(model_arch_and_sizes):
            return derive_subset_model_combinations(saved_combinations, model_arch_and_sizes)
        else:
            return saved_combinations

    # Otherwise generate new combinations
    assert seed is not None
    assert code_version is not None
    model_evaluations = get_model_evaluations(code_version, model_arch_and_sizes)
    # Get all prompts
    all_prompts = set(model_evaluations[model_arch_and_sizes[0]].index)

    # Create a DataFrame with correctness for each model
    correctness_df = pd.DataFrame(index=sorted(all_prompts))
    for model_arch_and_size in tqdm(model_arch_and_sizes, desc="Loading model evaluations"):
        model_df = model_evaluations[model_arch_and_size]
        correctness_df[model_arch_and_size] = [
            model_df.at[idx, COLS.EVALUATE_MODEL.MODEL_CORRECT] for idx in correctness_df.index
        ]

    combination_patterns = defaultdict(list)

    # Iterate over all prompts and group them by correctness pattern
    for prompt_idx in tqdm(correctness_df.index, desc="Processing prompts"):
        # Determine which models are correct and incorrect for this prompt
        correct_models = set()
        incorrect_models = set()

        for model in model_arch_and_sizes:
            if correctness_df.at[prompt_idx, model]:
                correct_models.add(model)
            else:
                incorrect_models.add(model)

        # Create a key that identifies this correctness pattern
        pattern_key = (frozenset(correct_models), frozenset(incorrect_models))
        combination_patterns[pattern_key].append(prompt_idx)

    # Generate the ModelCombination objects from the grouped prompts
    combinations = []
    for (correct_models, incorrect_models), prompts in tqdm(combination_patterns.items(), desc="Creating combinations"):
        random.seed(seed)
        chosen_prompt = random.choice(prompts)

        combinations.append(
            ModelCombination(
                correct_models=set(correct_models),
                incorrect_models=set(incorrect_models),
                prompts=prompts,
                chosen_prompt=chosen_prompt,
            )
        )

    save_model_combinations_prompts(combinations)
    return get_model_combinations_prompts(code_version, model_arch_and_sizes, seed)


def derive_subset_model_combinations(
    saved_combinations: list[ModelCombination],
    requested_models: list[MODEL_ARCH_AND_SIZE],
) -> list[ModelCombination]:
    """Derive model combinations for a subset using saved combinations with O(|C|) complexity."""
    requested_set = set(requested_models)
    pattern_map: dict[
        tuple[frozenset, frozenset],
        tuple[list[TPromptOriginalIndex], list[TPromptOriginalIndex]],
    ] = {}

    # First pass: Group by projected patterns and collect prompts
    for combo in tqdm(saved_combinations, desc="Projecting combinations"):
        # Project to subset - O(|S|) per combination
        proj_correct = frozenset(m for m in combo.correct_models if m in requested_set)
        proj_incorrect = frozenset(m for m in combo.incorrect_models if m in requested_set)

        # Get existing or create new entry
        key = (proj_correct, proj_incorrect)
        prompts, chosen_prompts = pattern_map.get(key, ([], []))

        # Extend with this combination's prompts
        prompts.extend(combo.prompts)
        if combo.chosen_prompt is not None:
            chosen_prompts.append(combo.chosen_prompt)

        pattern_map[key] = (prompts, chosen_prompts)

    # Second pass: Create new combinations
    new_combinations = []
    for (correct, incorrect), (prompts, chosen_prompts) in tqdm(
        pattern_map.items(), desc="Creating subset combinations"
    ):
        # Remove duplicate prompts while preserving order
        seen = set()
        unique_prompts = [p for p in prompts if not (p in seen or seen.add(p))]

        # Preserve chosen prompt order from original combinations
        chosen_candidates = [p for p in chosen_prompts if p in unique_prompts]
        chosen_prompt = chosen_candidates[0]

        new_combinations.append(
            ModelCombination(
                correct_models=set(correct),
                incorrect_models=set(incorrect),
                prompts=unique_prompts,
                chosen_prompt=chosen_prompt,
            )
        )

    return sorted(new_combinations, key=lambda x: (-len(x.prompts), x.chosen_prompt))
