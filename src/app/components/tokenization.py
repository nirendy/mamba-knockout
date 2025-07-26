import functools
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.app.components.prompt_filter import SelectPromptsComponent, show_prompt
from src.core.types import TErrorMessage, TokenType, TPromptOriginalIndex, TTokenIndex
from src.data_ingestion.data_defs.data_defs import (
    Prompts,
    Tokenizers,
    TUniqueTokenizerName,
    UniqueTokenizerInfo,
)
from src.data_ingestion.helpers.logits_utils import find_token_range
from src.utils.streamlit.components.aagrid import SelectionMode
from src.utils.streamlit.helpers.background_task import (
    TasksManager,
    TaskStatus,
    show_tasks_manager_summary,
)
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey


@dataclass
class PromptTokenizationStats:
    """Stores statistics for a single prompt tokenized by a single tokenizer."""

    token_count: Optional[int] = None
    # Stores count per token type, None if error during type count extraction
    token_type_counts: Dict[TokenType, Optional[int]] = field(default_factory=dict)
    # Stores the main tokenization error type, if any
    error_msg: Optional[TErrorMessage] = None
    # Stores error types encountered during specific token type extraction
    token_type_error_msg: Dict[TokenType, Optional[TErrorMessage]] = field(default_factory=dict)
    token_type_edges: Dict[TokenType, Optional[Tuple[TTokenIndex, TTokenIndex]]] = field(default_factory=dict)
    subject_token_edges: Optional[Tuple[TTokenIndex, TTokenIndex]] = None
    # Comparison between true_id and true_id_v2
    true_id_match: Optional[bool] = None  # True if both methods return same result, False if different, None if error
    true_id_error_msg: Optional[TErrorMessage] = None  # Error message if comparison failed
    true_id_values: Dict[str, Any] = field(default_factory=dict)  # Store actual values when they differ


@dataclass
class TokenizationResults:
    """Stores accumulated tokenization results in a nested structure."""

    # {tokenizer_name: {prompt_idx: PromptTokenizationStats}}
    data: Dict[TUniqueTokenizerName, Dict[TPromptOriginalIndex, PromptTokenizationStats]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Track completed pairs to avoid re-computation
    processed_pairs: Set[Tuple[TPromptOriginalIndex, TUniqueTokenizerName]] = field(default_factory=set)
    # Lock for thread-safe updates to shared data structures
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_stats(
        self, tokenizer_name: TUniqueTokenizerName, prompt_idx: TPromptOriginalIndex
    ) -> Optional[PromptTokenizationStats]:
        """Safely retrieve stats for a specific pair."""
        return self.data.get(tokenizer_name, {}).get(prompt_idx)

    # Helper methods to derive aggregate stats can be added here if needed by visualizer
    def get_total_errors_per_tokenizer(self) -> Dict[TUniqueTokenizerName, int]:
        errors = defaultdict(int)
        for tk_name, prompt_dict in self.data.items():
            for stats in prompt_dict.values():
                if stats.error_msg:
                    errors[tk_name] += 1
        return dict(errors)

    def get_failure_reasons_per_tokenizer(self) -> Dict[TUniqueTokenizerName, Dict[str, int]]:
        reasons = defaultdict(lambda: defaultdict(int))
        for tk_name, prompt_dict in self.data.items():
            for stats in prompt_dict.values():
                if stats.error_msg:
                    reasons[tk_name][stats.error_msg] += 1
                for tt_error in stats.token_type_error_msg.values():
                    if tt_error:
                        # Prefix to distinguish from main tokenization errors
                        reasons[tk_name][f"type_{tt_error}"] += 1
        return dict(reasons)


type TokenizationTaskInput = Tuple[UniqueTokenizerInfo, List[TPromptOriginalIndex], Prompts, TokenizationResults]


def tokenize_prompts_task(
    args: TokenizationTaskInput,
    cancellation_event: threading.Event,
) -> None:
    """Worker function for tokenizing a subset of prompts with one tokenizer."""
    tokenizer_info, prompt_indices_to_process, prompts, results = args
    tokenizer_name = tokenizer_info.display_name

    for prompt_idx in prompt_indices_to_process:
        if cancellation_event.is_set():
            return  # Stop processing if cancellation is requested

        prompt = prompts.get_prompt(prompt_idx)

        # Initialize stats for this specific pair
        stats = PromptTokenizationStats()

        try:
            # Tokenize the prompt
            input_ids = prompt.input_ids(tokenizer_info.tokenizer, "cpu")
            stats.token_count = len(input_ids[0])

            # Get token counts by type
            for token_type in TokenType:
                try:
                    token_indices = prompt.get_knockout_idx(tokenizer_info.tokenizer, token_type, "cpu")
                    stats.token_type_counts[token_type] = len(token_indices)
                    stats.token_type_edges[token_type] = (token_indices[0], token_indices[-1])
                except Exception as e_type:
                    stats.token_type_error_msg[token_type] = TErrorMessage(str(e_type))

            stats.subject_token_edges = find_token_range(tokenizer_info.tokenizer, input_ids[0], prompt.subject)

            # Compare true_id and true_id_v2
            try:
                # Get results from both methods
                true_id_tokens = prompt.true_id(tokenizer_info.tokenizer, "cpu")
                true_id_v2_tokens = prompt.true_id_v2(tokenizer_info.tokenizer, "cpu")

                # Compare the tensors
                same_shape = true_id_tokens.shape == true_id_v2_tokens.shape
                if same_shape:
                    tokens_match = torch.equal(true_id_tokens, true_id_v2_tokens)
                else:
                    tokens_match = False

                stats.true_id_match = tokens_match

                # If tokens don't match, store the actual values
                if not tokens_match:
                    stats.true_id_values = {
                        "true_id": {
                            "tokens": true_id_tokens.tolist(),
                            "decoded": tokenizer_info.tokenizer.decode(true_id_tokens[0]),
                            "shape": list(true_id_tokens.shape),
                        },
                        "true_id_v2": {
                            "tokens": true_id_v2_tokens.tolist(),
                            "decoded": tokenizer_info.tokenizer.decode(true_id_v2_tokens[0]),
                            "shape": list(true_id_v2_tokens.shape),
                        },
                        "prompt": prompt.prompt,
                        "true_word": prompt.true_word,
                    }
            except Exception as e_true_id:
                stats.true_id_match = None  # Mark as error
                stats.true_id_error_msg = TErrorMessage(str(e_true_id))

        except Exception as e_main:
            stats.error_msg = TErrorMessage(str(e_main))

        # --- Update shared results safely ---
        with results.lock:
            # Store the completed stats object
            results.data[tokenizer_name][prompt_idx] = stats
            # Mark this specific pair as processed
            results.processed_pairs.add((prompt_idx, tokenizer_name))


class AnalysisJob(TasksManager[TokenizationTaskInput, None]):
    """Manages the state of a tokenization analysis task."""

    def __init__(self, prompts: Prompts, unique_tokenizers: Tokenizers):
        """Initialize the analysis job with prompts and tokenizers."""
        super().__init__()  # Initialize the parent TasksManager
        self.prompts = prompts
        self.unique_tokenizers = unique_tokenizers

    def __hash__(self) -> int:
        prompt_ids_hash = hash(tuple(sorted(self.prompts.keys())))
        tokenizer_names_hash = hash(tuple(sorted([t.display_name for t in self.unique_tokenizers])))
        return hash((prompt_ids_hash, tokenizer_names_hash))

    @property
    def total_pairs_needed(self) -> int:
        return len(self.prompts) * len(self.unique_tokenizers)

    def missing_pair_per_tokenizer(
        self, existing_results: TokenizationResults
    ) -> Dict[UniqueTokenizerInfo, List[TPromptOriginalIndex]]:
        return {
            tk_info: list(set(self.prompts.keys()) - set(existing_results.data.get(tk_info.display_name, {}).keys()))
            for tk_info in self.unique_tokenizers
        }

    def cancel(self):
        self.cancel_all_tasks()  # Use parent method directly

    def start(self, existing_results: TokenizationResults):
        missing_pairs = self.missing_pair_per_tokenizer(existing_results)

        if not any(missing_pairs.values()):
            return  # No work to do if no missing pairs

        # Create tasks for each tokenizer
        for tokenizer_info, prompt_indices in missing_pairs.items():
            if not prompt_indices:  # Skip empty prompt lists
                continue

            task_name = tokenizer_info.display_name

            # Create input data tuple for the task
            task_input = (tokenizer_info, prompt_indices, self.prompts, existing_results)

            # Create and start the background task
            self.create_and_start_task(task_name, tokenize_prompts_task, task_input)


@dataclass
class TokenizationComponentCache:
    """Cache for tokenization results and ongoing tasks."""

    results: TokenizationResults = field(default_factory=TokenizationResults)
    task: AnalysisJob = field(
        default_factory=lambda: (
            AnalysisJob(
                prompts=Prompts({}),
                unique_tokenizers=Tokenizers([]),
            )
        )
    )

    def analyze_tokenization(self, prompts: Prompts, unique_tokenizers: Tokenizers, force_refresh: bool = False):
        """Create and start a new analysis job if needed or requested."""
        # Create a new task
        new_task = AnalysisJob(
            prompts=prompts,
            unique_tokenizers=unique_tokenizers,
        )

        # Only start if it's a new task configuration or force refresh is requested
        if hash(new_task) != hash(self.task) or force_refresh:
            self.task.cancel()
            self.task = new_task
            self.task.start(self.results)

        return self.task

    def get_task_subset_results(self) -> TokenizationResults:
        """Returns a *new* TokenizationResults object containing only the data relevant to the current task's scope."""
        subset = TokenizationResults()
        task_prompts = set(self.task.prompts.keys())
        task_tokenizers = {t.display_name for t in self.task.unique_tokenizers}

        with self.results.lock:  # Lock during read to ensure consistency
            subset.processed_pairs = set(
                (p_idx, tk_name)
                for p_idx, tk_name in self.results.processed_pairs
                if p_idx in task_prompts and tk_name in task_tokenizers
            )

            for tk_name in task_tokenizers:
                if tk_name not in self.results.data:
                    continue
                for p_idx, stats in self.results.data[tk_name].items():
                    if p_idx in task_prompts:
                        # Ensure nested dict exists before assignment
                        if tk_name not in subset.data:
                            subset.data[tk_name] = {}
                        subset.data[tk_name][p_idx] = stats  # Shallow copy of stats is okay

        return subset


class TokenizationResultsVisualizer(StreamlitComponent):
    """Component for visualizing tokenization analysis results."""

    def __init__(self, results: TokenizationResults):
        self.results = results

    def render(self) -> None:
        """Render the tokenization results visualization."""
        if not self.results:
            st.info("No results object available to display.")
            return

        # Use tabs for better organization of the analysis sections
        analysis_tabs = st.tabs(
            [
                "📊 Overall Statistics",
                "❌ Error Analysis",
                "📏 Token Counts",
                "🔄 Token Type Analysis",
                "🔍 Disagreement Analysis",
                "⚖️ true_id Comparison",
            ]
        )

        # 1. Overall statistics
        with analysis_tabs[0]:
            self._render_overall_statistics()

        # 2. Error analysis
        with analysis_tabs[1]:
            self._render_error_analysis()

        # 3. Token count distribution
        with analysis_tabs[2]:
            self._render_token_count_distribution()

        # 4. Token type analysis
        with analysis_tabs[3]:
            self._render_token_type_analysis()

        # 5. Tokenization disagreement analysis
        with analysis_tabs[4]:
            self._render_disagreement_analysis()

        # 6. true_id comparison analysis
        with analysis_tabs[5]:
            self._render_true_id_comparison()

    def _render_overall_statistics(self):
        """Render overall statistics section based on the new structure."""
        col1, col2 = st.columns(2)
        with col1:
            pass

        with col2:
            # Calculate stats based on processed pairs in the subset passed to the visualizer
            total_processed = len(self.results.processed_pairs)
            total_errors = sum(self.results.get_total_errors_per_tokenizer().values())  # Use helper

            success_rate = ((total_processed - total_errors) / total_processed) * 100 if total_processed > 0 else 0
            st.metric("Processed Prompt-Tokenizer Pairs", total_processed)
            st.metric(
                "Tokenization Success Rate (Processed)",
                f"{success_rate:.2f}% ({total_processed - total_errors}/{total_processed})",
            )
            st.metric("Tokenization Errors Encountered", total_errors)

    def _render_error_analysis(self):
        """Render error analysis section using helper methods."""
        errors_per_tokenizer = self.results.get_total_errors_per_tokenizer()
        total_errors = sum(errors_per_tokenizer.values())

        if total_errors > 0:
            # Errors per tokenizer
            error_data_list = [
                {"Tokenizer": name, "Error Count": count} for name, count in errors_per_tokenizer.items() if count > 0
            ]
            if error_data_list:
                error_data = pd.DataFrame(error_data_list).sort_values("Error Count", ascending=False)
                fig = px.bar(error_data, x="Tokenizer", y="Error Count", title="Tokenization Errors by Tokenizer")
                st.plotly_chart(fig)
            else:
                st.info("No tokenization errors recorded for specific tokenizers in this subset.")

            # Display error types (failure reasons)
            with st.expander("Failure Reason Details"):
                failure_reasons = self.results.get_failure_reasons_per_tokenizer()
                error_types_data = []
                for tokenizer, errors in failure_reasons.items():
                    for error_type, count in errors.items():
                        if count > 0:
                            error_types_data.append({"Tokenizer": tokenizer, "Error Type": error_type, "Count": count})

                if error_types_data:
                    error_types_df = pd.DataFrame(error_types_data)
                    fig = px.bar(
                        error_types_df,
                        x="Tokenizer",
                        y="Count",
                        color="Error Type",
                        title="Failure Reasons by Tokenizer",
                        barmode="stack",
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("No specific failure reasons recorded in this subset.")
        else:
            st.success("No tokenization errors found in this result subset!")

    def _render_token_count_distribution(self):
        """Render token count distribution section from the new structure."""
        token_count_data = []
        for tokenizer_name in self.results.data.keys():
            valid_counts = []
            prompt_data = self.results.data.get(tokenizer_name, {})
            for stats in prompt_data.values():
                # Check if prompt is in the current subset (redundant if visualizer receives subset results)
                # if prompt_idx in self.results.prompt_idxs:
                if stats.token_count is not None and stats.token_count > 0:
                    valid_counts.append(stats.token_count)

            if valid_counts:
                series = pd.Series(valid_counts)
                token_count_data.append(
                    {
                        "Tokenizer": tokenizer_name,
                        "Average Token Count": series.mean(),
                        "Min Token Count": series.min(),
                        "Max Token Count": series.max(),
                        "Std Dev": series.std(),
                        "Processed Count": len(valid_counts),  # How many prompts were successfully tokenized
                    }
                )

        if token_count_data:
            token_count_df = pd.DataFrame(token_count_data)
            # Bar chart for average token counts
            fig = px.bar(
                token_count_df,
                x="Tokenizer",
                y="Average Token Count",
                error_y="Std Dev",
                title="Average Token Count by Tokenizer (Excluding Errors)",
            )
            st.plotly_chart(fig)
            # Table view in expander
            with st.expander("Token Count Statistics (Table View)"):
                st.dataframe(token_count_df)
        else:
            st.info("No valid token count data available to display in this subset.")

    def _render_token_type_analysis(self):
        """Render token type analysis section from the new structure."""
        valid_token_types = list(TokenType)  # Use all defined token types

        if not valid_token_types:
            st.warning("No TokenTypes defined.")
            return

        # Check if any data exists for any token type
        has_data = any(self.results.data.values())
        if not has_data:
            st.info("No tokenization data has been generated yet.")
            return

        selected_source = st.selectbox(
            "Select Source Token Type", valid_token_types, format_func=lambda x: str(x), key="tt_source_select"
        )
        selected_target = st.selectbox(
            "Select Target Token Type", valid_token_types, format_func=lambda x: str(x), key="tt_target_select"
        )

        if not selected_source or not selected_target:
            st.warning("Please select both source and target token types.")
            return

        st.info(f"Analysis for Source Type: {selected_source} and Target Type: {selected_target}")

        source_data = []
        target_data = []
        ratio_data = []

        for tokenizer_name in self.results.data.keys():
            source_counts_valid = []
            target_counts_valid = []
            ratios_valid = []

            prompt_data = self.results.data.get(tokenizer_name, {})
            for prompt_idx, stats in prompt_data.items():
                # Check if prompt is in the current subset (redundant if visualizer receives subset results)
                # if prompt_idx in self.results.prompt_idxs:
                source_count = stats.token_type_counts.get(selected_source)
                target_count = stats.token_type_counts.get(selected_target)

                # Filter out entries where either count is None or source count is <= 0
                if source_count is not None and target_count is not None and source_count > 0:
                    source_counts_valid.append(source_count)
                    target_counts_valid.append(target_count)
                    ratios_valid.append(target_count / source_count)

            if source_counts_valid:  # Check if we found any valid pairs for this tokenizer
                source_series = pd.Series(source_counts_valid)
                target_series = pd.Series(target_counts_valid)
                ratio_series = pd.Series(ratios_valid)

                source_data.append(
                    {
                        "Tokenizer": tokenizer_name,
                        "Average Count": source_series.mean(),
                        "Std Dev": source_series.std(),
                        "Type": str(selected_source),
                    }
                )
                target_data.append(
                    {
                        "Tokenizer": tokenizer_name,
                        "Average Count": target_series.mean(),
                        "Std Dev": target_series.std(),
                        "Type": str(selected_target),
                    }
                )
                ratio_data.append(
                    {
                        "Tokenizer": tokenizer_name,
                        "Average Ratio": ratio_series.mean(),
                        "Std Dev Ratio": ratio_series.std(),
                        "Ratio Type": f"{selected_target}/{selected_source}",
                    }
                )

        # Use columns for better layout
        col1, col2 = st.columns(2)
        # Combine source and target data for a grouped bar chart
        with col1:
            combined_data = source_data + target_data
            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                fig = px.bar(
                    combined_df,
                    x="Tokenizer",
                    y="Average Count",
                    error_y="Std Dev",
                    color="Type",
                    title="Average Token Counts by Type (Excluding Errors)",
                    barmode="group",
                )
                st.plotly_chart(fig)
            else:
                st.info(f"No valid data found for types {selected_source} or {selected_target} in this subset.")
        # Show ratio data
        with col2:
            if ratio_data:
                ratio_df = pd.DataFrame(ratio_data)
                fig = px.bar(
                    ratio_df,
                    x="Tokenizer",
                    y="Average Ratio",
                    error_y="Std Dev Ratio",
                    title=f"Ratio of {selected_target} to {selected_source} Tokens",
                )
                st.plotly_chart(fig)
            else:
                st.info(f"No valid ratio data found for {selected_target}/{selected_source} in this subset.")

    def _render_disagreement_analysis(self):
        """Render tokenization disagreement analysis section from the new structure."""
        if len(self.results.data) < 2:
            st.info("Need at least 2 tokenizers in the results to analyze disagreement.")
            return

        if not self.results.processed_pairs:
            st.info("No prompts processed yet to analyze disagreement.")
            return

        # Create a progress bar for variance calculation
        with st.spinner("Calculating tokenization disagreement metrics..."):
            token_count_variance = []
            # Iterate through prompts relevant to the current subset
            for prompt_idx, _ in self.results.processed_pairs:
                counts = []
                # Iterate through tokenizers relevant to the current subset
                for tokenizer_name in self.results.data.keys():
                    stats = self.results.get_stats(tokenizer_name, prompt_idx)
                    if stats and stats.token_count is not None and stats.token_count > 0:
                        counts.append(stats.token_count)

                if len(counts) > 1:  # Need at least 2 successful tokenizations for this prompt
                    series = pd.Series(counts)
                    variance = series.var()
                    token_count_variance.append(
                        {
                            "Prompt Index": prompt_idx,
                            "Token Count Variance": variance,
                            "Min Count": series.min(),
                            "Max Count": series.max(),
                            "Range": series.max() - series.min(),
                            "Num Tokenizers": len(counts),  # How many tokenizers provided a valid count for this prompt
                        }
                    )

        if token_count_variance:
            variance_df = pd.DataFrame(token_count_variance)
            variance_df = variance_df.sort_values("Token Count Variance", ascending=False)
            # Show prompts with highest disagreement
            st.subheader("Prompts with Highest Tokenization Disagreement")
            top_n = min(10, len(variance_df))
            st.dataframe(variance_df.head(top_n))
            # Use columns for better layout
            col1, col2 = st.columns(2)
            # Histogram of token count variance
            with col1:
                fig = px.histogram(
                    variance_df,
                    x="Token Count Variance",
                    nbins=30,
                    title="Distribution of Token Count Variance Across Prompts",
                )
                st.plotly_chart(fig)
            # Scatter plot of variance vs range
            with col2:
                fig = px.scatter(
                    variance_df,
                    x="Range",
                    y="Token Count Variance",
                    hover_data=["Prompt Index", "Num Tokenizers"],
                    title="Variance vs Range in Token Counts Per Prompt",
                    opacity=0.6,
                )
                st.plotly_chart(fig)
        else:
            st.info(
                "Not enough comparable data across tokenizers for the prompts in this subset to calculate disagreement."
            )

    def _render_true_id_comparison(self):
        """Render a detailed comparison between true_id and true_id_v2 methods across tokenizers.

        This tab analyzes whether the two methods for generating true token IDs produce the same results:
        1. true_id: directly tokenizes the true_word
        2. true_id_v2: computes the difference between prompt and prompt+true_word tokenization

        The analysis includes:
        - Mapping of tokenizers to their associated models
        - Statistics on matching vs differing cases per tokenizer
        - Visualization of comparison results
        - Detailed examples of cases where the methods produce different results
        """
        if not self.results.processed_pairs:
            st.info("No prompts processed yet to compare true_id.")
            return

        # Add analysis for tokenizers with differences
        st.subheader("true_id vs true_id_v2 Comparison")

        @dataclass
        class TrueIdComparisonStats:
            matching: int = 0
            different: int = 0
            errors: int = 0
            unique_errors: Set[TErrorMessage] = field(default_factory=set)
            total: int = 0

        # First, gather stats on matching vs differing cases per tokenizer
        match_stats = defaultdict(TrueIdComparisonStats)
        diff_examples = defaultdict(list)

        # Process all prompt/tokenizer combinations
        for tokenizer_name, prompts_dict in self.results.data.items():
            for prompt_idx, stats in prompts_dict.items():
                match_stats[tokenizer_name].total += 1

                if stats.true_id_match is None:
                    match_stats[tokenizer_name].errors += 1
                    assert stats.true_id_error_msg is not None
                    match_stats[tokenizer_name].unique_errors.add(stats.true_id_error_msg)
                elif stats.true_id_match:
                    match_stats[tokenizer_name].matching += 1
                else:
                    match_stats[tokenizer_name].different += 1
                    # Store examples where they differ
                    if stats.true_id_values:
                        diff_examples[tokenizer_name].append({"prompt_idx": prompt_idx, "values": stats.true_id_values})

        # Convert to DataFrame for display
        if not match_stats:
            st.info("No true_id comparison data available yet.")
            return

        match_stats_list = []
        for tokenizer, counts in match_stats.items():
            match_stats_list.append(
                {
                    "Tokenizer": tokenizer,
                    "Matching": counts.matching,
                    "Different": counts.different,
                    "Errors": counts.errors,
                    "Unique Errors": len(counts.unique_errors),
                    "Total": counts.total,
                    "Match Rate (%)": round(counts.matching / counts.total * 100, 2) if counts.total > 0 else 0,
                }
            )

        match_stats_df = pd.DataFrame(match_stats_list)
        st.write("### true_id Match Statistics by Tokenizer")
        st.dataframe(match_stats_df.sort_values("Match Rate (%)", ascending=False))

        # Visualize the comparison
        fig = px.bar(
            match_stats_df,
            x="Tokenizer",
            y=["Matching", "Different", "Errors"],
            title="true_id vs true_id_v2 Comparison Results by Tokenizer",
            barmode="stack",
        )
        st.plotly_chart(fig)

        # Show examples where true_id and true_id_v2 differ
        st.write("### Examples Where true_id and true_id_v2 Differ")

        # Let user select a tokenizer to see examples
        tokenizers_with_diffs = [tk for tk, examples in diff_examples.items() if examples]

        if tokenizers_with_diffs:
            selected_tokenizer = st.selectbox(
                "Select Tokenizer to See Examples", tokenizers_with_diffs, key="diff_tokenizer_select"
            )

            for i, example in enumerate(diff_examples[selected_tokenizer]):
                with st.expander(f"Example {i + 1}: Prompt {example['prompt_idx']}"):
                    values = example["values"]
                    # Display prompt and true_word
                    st.write(f"**Prompt:** {values['prompt']}")
                    st.write(f"**True Word:** {values['true_word']}")

                    # Display true_id results
                    st.write("**true_id results:**")
                    st.json(
                        {
                            "tokens": values["true_id"]["tokens"],
                            "decoded": values["true_id"]["decoded"],
                            "shape": values["true_id"]["shape"],
                        }
                    )

                    # Display true_id_v2 results
                    st.write("**true_id_v2 results:**")
                    st.json(
                        {
                            "tokens": values["true_id_v2"]["tokens"],
                            "decoded": values["true_id_v2"]["decoded"],
                            "shape": values["true_id_v2"]["shape"],
                        }
                    )
        else:
            st.info("No examples found where true_id and true_id_v2 differ for any tokenizer.")

        # if stats.true_id_match is not None:
        #     st.write("### true_id vs true_id_v2 Comparison")
        #     if stats.true_id_match:
        #         st.success("✅ true_id and true_id_v2 methods produce matching results")
        #     else:
        #         st.error("❌ true_id and true_id_v2 methods produce different results")

        #         if stats.true_id_values:
        #             values = stats.true_id_values
        #             col1, col2 = st.columns(2)

        #             with col1:
        #                 st.write("**true_id results:**")
        #                 st.json(
        #                     {
        #                         "tokens": values["true_id"]["tokens"],
        #                         "decoded": values["true_id"]["decoded"],
        #                         "shape": values["true_id"]["shape"],
        #                     }
        #                 )

        #             with col2:
        #                 st.write("**true_id_v2 results:**")
        #                 st.json(
        #                     {
        #                         "tokens": values["true_id_v2"]["tokens"],
        #                         "decoded": values["true_id_v2"]["decoded"],
        #                         "shape": values["true_id_v2"]["shape"],
        #                     }
        #                 )
        # elif stats.true_id_error_msg:
        #     st.error(f"Error during true_id comparison: {stats.true_id_error_msg}")


class SinglePromptAnalyzer(StreamlitComponent):
    """Component for analyzing and displaying tokenization of a single prompt using cached results."""

    def __init__(
        self,
        prompts: Prompts,  # Still need original prompts for display
        unique_tokenizers: Tokenizers,
        results_cache: TokenizationResults,  # Pass the global results cache
    ):
        self.prompts = prompts
        self.unique_tokenizers = unique_tokenizers
        self.results_cache = results_cache

    def render(self) -> Optional[TPromptOriginalIndex]:
        """Render the single prompt analysis component by fetching cached stats."""
        container = st.container()

        with container:
            if not self.prompts:
                st.warning("No prompts available for analysis.")
                return None

            # Let user select a prompt
            grid_results = SelectPromptsComponent(
                self.prompts.to_df(),
                SelectionMode.SINGLE,
                key=f"tokenization_prompt_selection_{hash(self.prompts)}",
            ).render()

            if grid_results is None or grid_results.empty:
                st.info("Select a prompt from the table above to see detailed tokenization.")
                return None

            # Get selected prompt
            selected_prompt_row = grid_results.iloc[0]
            prompt_idx = TPromptOriginalIndex(int(selected_prompt_row["original_idx"]))
            selected_prompt_new = self.prompts.get_prompt(prompt_idx)

            if not selected_prompt_new:
                st.error(f"Selected prompt index {prompt_idx} not found.")
                return None

            # Show the original prompt
            with st.expander("Original Prompt", expanded=True):
                show_prompt(selected_prompt_new)  # Use the PromptNew object

            st.subheader("Tokenization Details (from cached results)")

            # Iterate through available tokenizers
            if not self.unique_tokenizers:
                st.warning("No tokenizers available for analysis.")
                return prompt_idx

            for tokenizer_info in self.unique_tokenizers:
                tokenizer_name = tokenizer_info.display_name
                label = (
                    f"Tokenizer: {tokenizer_name}"
                    f" | {len(tokenizer_info.model_arch_and_sizes)} models: {tokenizer_info.model_arch_and_sizes}"
                )
                with st.expander(label, expanded=False):  # Start collapsed
                    # Fetch stats from the shared cache
                    with self.results_cache.lock:  # Access cache safely
                        stats = self.results_cache.get_stats(tokenizer_name, prompt_idx)

                    if stats:
                        # Display fetched stats
                        st.metric("Token Count", stats.token_count if stats.token_count is not None else "Error")

                        if stats.error_msg:
                            st.error(f"Tokenization Error: {stats.error_msg}")
                        else:
                            st.success("Tokenization Successful")

                        # Display token type counts
                        st.write("Token Type Counts:")
                        type_counts_data = []
                        for token_type, count in stats.token_type_counts.items():
                            status = "OK"
                            error_detail = stats.token_type_error_msg.get(token_type)
                            if count is None:
                                status = f"Error ({error_detail or 'Unknown'})"
                                count_display = "N/A"
                            else:
                                count_display = str(count)
                            type_counts_data.append({"Type": str(token_type), "Count": count_display, "Status": status})

                        if type_counts_data:
                            st.dataframe(pd.DataFrame(type_counts_data))
                        else:
                            st.info("No token type counts available.")

                        # Optionally, re-tokenize *only for display* if needed, but avoid analysis
                        # This adds overhead but provides the visual token breakdown
                        try:
                            prompt_obj = selected_prompt_new  # Get Prompt object
                            input_ids = prompt_obj.input_ids(tokenizer_info.tokenizer, "cpu")
                            tokens = [tokenizer_info.tokenizer.decode([token_id.item()]) for token_id in input_ids[0]]
                            token_display = " | ".join([f"{i}: '{t}'" for i, t in enumerate(tokens)])
                            st.text_area(
                                "Tokenization Display",
                                token_display,
                                height=100,
                                key=f"token_display_{tokenizer_name}_{prompt_idx}",
                                disabled=True,
                            )
                        except Exception as display_e:
                            st.warning(f"Could not display token breakdown: {display_e}")

                    else:
                        st.info("Analysis results not yet available for this prompt/tokenizer pair. Refresh results.")

        return prompt_idx


class TokenizationVisualizerComponent(StreamlitComponent):
    """Main component for visualizing tokenization of prompts across different tokenizers."""

    def __init__(
        self,
        prompts: Prompts,
        unique_tokenizers: Tokenizers,
        cache_key_name: str = "tokenization_component_cache",
    ):
        self.prompts = prompts
        self.unique_tokenizers = unique_tokenizers
        self.cache_sk = SessionKey[TokenizationComponentCache](cache_key_name, TokenizationComponentCache())
        self.cache_sk.init_default()

    def render(self) -> None:
        """Render the tokenization visualization component."""
        st.write("# Tokenization Analysis")

        # Add tabs for different analysis modes with emojis for better visual cues
        tabs = st.tabs(["📊 Compare All Prompts", "🔍 Analyze Single Prompt"])

        # --- Tab 1: Compare All Prompts ---
        with tabs[0]:
            st.subheader("Compare All Prompts")
            # Initialize containers

            # --- Trigger Analysis ---
            cache = self.cache_sk.value
            cache.analyze_tokenization(
                self.prompts,
                self.unique_tokenizers,
            )

            if cache.task.status != TaskStatus.COMPLETED:
                # Task progress and status display

                def get_additional_metrics(task: AnalysisJob) -> Tuple[Dict[str, str], float]:
                    missing_amount = functools.reduce(
                        lambda acc, missing_prompts: acc + len(missing_prompts),
                        task.missing_pair_per_tokenizer(cache.results).values(),
                        0,
                    )
                    total_pairs_in_task = task.total_pairs_needed
                    percent_complete = (total_pairs_in_task - missing_amount) / total_pairs_in_task

                    # Additional metrics for task manager summary
                    return (
                        {
                            "Total Pairs": str(total_pairs_in_task),
                            "Remaining": str(missing_amount),
                        },
                        percent_complete,
                    )

                # Show task manager summary with controls
                show_tasks_manager_summary(
                    lambda: cache.task,
                    on_start_click=lambda: cache.analyze_tokenization(
                        self.prompts, self.unique_tokenizers, force_refresh=True
                    ),
                    get_additional_metrics=get_additional_metrics,
                    button_keys_prefix="task_viz_",
                )

            col1, col2 = st.columns([8, 2])
            with col1:
                show_analysis = st.checkbox("Show Comparison Analysis", value=True, key="show_analysis_check")
            with col2:
                if st.button("🧹 Clear Results", key="clear_results"):
                    self.cache_sk.reset_value()
                    st.rerun()

            if show_analysis:
                # Pass the subset results relevant to the *current* task
                subset_results = cache.get_task_subset_results()
                TokenizationResultsVisualizer(subset_results).render()
            else:
                st.info("Comparison Analysis hidden. Check the box above to show it.")

        # --- Tab 2: Analyze Single Prompt ---
        with tabs[1]:
            st.subheader("Analyze Single Prompt")
            # Pass the global results cache for lookup
            SinglePromptAnalyzer(self.prompts, self.unique_tokenizers, cache.results).render()
