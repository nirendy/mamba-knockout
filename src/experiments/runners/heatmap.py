"""
HeatmapExperiment: Single prompt-level experiment showing how different layers affect the model's token predictions

In this experiment implementation:
The sub-task is a prompt index ( notice - this experiment is not standard, we are not iterating over the dataset)
The inner loop is masking a sliding window over the model layers
The sub task result is a heatmap of the token probabilities for each layer in the window
The combined result is a dictionary of prompt index -> heatmap

"""

import functools
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable, ClassVar, Optional, TypedDict, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.analysis.plots.heatmaps import simple_diff_fixed
from src.core.names import ExperimentName
from src.core.types import (
    FeatureCategory,
    IHeatmap,
    TPromptOriginalIndex,
    TWindow,
    TWindowSize,
)
from src.data_ingestion.helpers.logits_utils import Prompt, decode_tokens, get_prompt_row_index
from src.experiments.infrastructure.base_runner import (
    BASE_OUTPUT_KEYS,
    BaseRunner,
    BaseVariantParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner


class HEATMAP_PLOT_FUNCS(StrEnum):
    _simple_diff_fixed_0_3 = "_simple_diff_fixed_0.3"


plot_suffix_to_function: dict[HEATMAP_PLOT_FUNCS, Callable] = {
    HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3: functools.partial(simple_diff_fixed),
}


@dataclass(frozen=True)
class HeatmapParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.heatmap)
    window_size: TWindowSize


class HeatmapDependencies(TypedDict):
    evaluate_model: EvaluateModelRunner


@dataclass
class HDF5HeatmapFile:
    path: Path

    def get_existing_prompt_idx(self) -> list[TPromptOriginalIndex]:
        with h5py.File(self.path, "r") as f:
            return [TPromptOriginalIndex(int(p)) for p in f.keys()]

    def get_prompt_idx_heatmaps(
        self, prompt_idx: Optional[list[TPromptOriginalIndex]] = None
    ) -> dict[TPromptOriginalIndex, pd.DataFrame]:
        result = {}
        with h5py.File(self.path, "r") as f:
            if prompt_idx is None:
                prompt_idx = [TPromptOriginalIndex(int(p)) for p in f.keys()]

            for p in prompt_idx:
                # Extract dataset as numpy array explicitly before converting to DataFrame
                dataset = f[str(p)]
                if isinstance(dataset, h5py.Dataset):
                    numpy_array = dataset[:]
                    result[TPromptOriginalIndex(int(p))] = pd.DataFrame(numpy_array)
        return result


HeatmapExperimentOutput = dict[TPromptOriginalIndex, IHeatmap]


@dataclass(frozen=True)
class HeatmapRunner(BaseRunner[HeatmapParams]):
    """Configuration for heatmap generation."""

    variant_params: HeatmapParams

    @staticmethod
    def _get_variant_params():
        return HeatmapParams

    @classmethod
    def get_variant_output_keys(cls):
        return super().get_variant_output_keys() + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
        ]

    @property
    def output_hdf5_path(self) -> HDF5HeatmapFile:
        """Return the path to the HDF5 file containing all prompt heatmaps."""
        return HDF5HeatmapFile(self.variation_paths.outputs_path / "heatmaps.h5")

    def get_remaining_prompt_original_indices(self):
        """Return the list of prompt indices that need to be computed."""
        if not self.output_hdf5_path.path.exists() or self.metadata_params.overwrite_existing_outputs:
            return self.input_params.filteration.contextualize(self).get_prompt_ids()

        existing_prompts = self.output_hdf5_path.get_existing_prompt_idx()
        return [
            idx
            for idx in self.input_params.filteration.contextualize(self).get_prompt_ids()
            if idx not in existing_prompts
        ]

    def get_outputs(self) -> HeatmapExperimentOutput:
        """Load all prompt heatmaps from the HDF5 file."""
        if not self.output_hdf5_path.path.exists():
            return {}

        return self.output_hdf5_path.get_prompt_idx_heatmaps(
            self.input_params.filteration.contextualize(self).get_prompt_ids()
        )

    def get_plot_output_path(self, prompt_idx: TPromptOriginalIndex, plot_name: HEATMAP_PLOT_FUNCS) -> Path:
        return self.variation_paths.plots_path / f"idx={prompt_idx}{plot_name}.png"

    def plot(self, plot_name: HEATMAP_PLOT_FUNCS, show: bool = False) -> None:
        plot(self, plot_name, show=show)

    def _compute_impl(self) -> None:
        run(self)

    def is_computed(self) -> bool:
        """Check if all required prompt heatmaps exist in the HDF5 file."""
        if not self.output_hdf5_path.path.exists():
            return False

        existing_prompts = self.output_hdf5_path.get_existing_prompt_idx()
        return all(
            idx in existing_prompts for idx in self.input_params.filteration.contextualize(self).get_prompt_ids()
        )

    def get_runner_dependencies(self) -> HeatmapDependencies:  # type: ignore
        return HeatmapDependencies(
            evaluate_model=EvaluateModelRunner.init_from_runner(
                self,
                variant_params=EvaluateModelParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                ),
            ),
        )


def plot(args: HeatmapRunner, plot_name: HEATMAP_PLOT_FUNCS, show: bool = False) -> None:
    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()
    tokenizer = args.variant_params.get_tokenizer
    model_id = args.variant_params.model_id

    prob_mats = args.get_outputs()
    for prompt_idx, prob_mat in tqdm(prob_mats.items(), desc="Plotting heatmaps"):
        prompt = get_prompt_row_index(data, prompt_idx)
        input_ids = prompt.input_ids(tokenizer, "cpu")
        toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
        last_tok = toks[-1]
        toks[-1] = toks[-1] + "*"

        fig, _ = simple_diff_fixed(
            prob_mat=prob_mat,
            model_id=model_id,
            window_size=args.variant_params.window_size,
            last_tok=last_tok,
            base_prob=prompt.base_prob,
            target_rank=prompt.target_rank,
            true_word=prompt.true_word,
            toks=toks,
        )
        if show:
            plt.show()
        else:
            output_path = args.get_plot_output_path(prompt.original_idx, HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)


def run(args: HeatmapRunner):
    print(args)
    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()
    remaining_idx = args.get_remaining_prompt_original_indices()
    if not remaining_idx:
        print("All heatmaps already exist")
        return

    args.create_experiment_dir()
    model_interface = args.variant_params.get_model_interface()
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = model_interface.n_layers()

    def forward_eval(prompt: Prompt, window: TWindow):
        true_id = prompt.true_id(tokenizer, "cpu")
        input_ids = prompt.input_ids(tokenizer, device)

        last_idx = input_ids.shape[1] - 1
        probs = np.zeros((input_ids.shape[1]))

        for idx in range(input_ids.shape[1]):
            num_to_masks = {layer: [(last_idx, idx)] for layer in window}

            next_token_probs = model_interface.generate_logits(
                input_ids=input_ids,
                num_to_masks=num_to_masks,
                feature_category=FeatureCategory.ALL,
            )
            probs[idx] = next_token_probs[0, true_id[:, 0]]
            torch.cuda.empty_cache()
        return probs

    windows = [
        TWindow(list(range(i, i + args.variant_params.window_size)))
        for i in range(0, n_layers - args.variant_params.window_size + 1)
    ]

    # Prepare HDF5 file for storing all heatmaps
    output_path = args.output_hdf5_path.path

    # Create or open the HDF5 file in append mode
    with h5py.File(output_path, "a") as hf:
        for prompt_idx in tqdm(remaining_idx, desc="Prompts"):
            prob_mat = []
            prompt = get_prompt_row_index(data, prompt_idx)

            for window in windows:
                model_interface.setup(layers=window)
                prob_mat.append(forward_eval(prompt, window))

            prob_mat_array = np.array(prob_mat).T

            # Store heatmap data in HDF5 file with prompt index as the key
            prompt_key = str(prompt.original_idx)
            if prompt_key in hf:
                del hf[prompt_key]  # Replace existing dataset if it exists

            hf.create_dataset(prompt_key, data=prob_mat_array)
