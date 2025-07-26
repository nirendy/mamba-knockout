# ruff: noqa: E731
class COMMON_TEXTS:
    error_details = "Show Error Details"
    preparing_to_run = "Preparing to run"
    processing_status = lambda current, total, run_what: f"Processed {current} / {total} {run_what}..."
    success_status = lambda count, run_what: f"Successfully submitted {count} {run_what}..."
    error_status = lambda count, run_what: f"Failed to submit {count} {run_what}..."
    submit_failed = lambda prompt_idx, error, run_what: f"Failed to submit job for {prompt_idx} {run_what}: {error}"
    LOADING = lambda what: f"Loading {what}..."


def show_filter_results_text(filtered_count: int, all_count: int) -> str:
    return f"{filtered_count} / {all_count} ({filtered_count / all_count * 100}%)"


class HOME_TEXTS:
    icon = "🔧"
    title = "Global Config"


class HEATMAP_TEXTS:
    icon = "🔥"
    title = "Heatmap Creation"
    description = "Create heatmaps for the selected prompts"

    # Headers
    MODEL_COMBINATIONS_HEADER = "Model Combinations Analysis"
    TAB_SELECT_COMBINATION = "Selected Combination"
    TAB_HEATMAP_GENERATION = "Heatmap Generation"
    TAB_HEATMAP_PLOTS_GENERATION = "Heatmap Plots Generation"
    POSSIBLE_PROMPTS_HEADER = "Possible Prompts"
    MODEL_COMBINATIONS_FILTERING = "Model Combinations Filtering"

    # Buttons
    show_combination = "Show Combination"
    show_prompts = "Show Prompts"

    # Other
    NO_SELECTED_COMBINATION = "Select a combination to continue"
    matching_counts = lambda len_filtered_prompts, len_all_prompts: (
        f"Found {show_filter_results_text(len_filtered_prompts, len_all_prompts)} prompts matching all criteria"
    )
    active_models_str = lambda active_models: " ∩ ".join(active_models)
    run_selected_prompts = lambda count: f"🚀 Run {count} Selected Jobs"
    processing_status = lambda current, total: COMMON_TEXTS.processing_status(current, total, "jobs")
    success_status = lambda count: COMMON_TEXTS.success_status(count, "jobs")
    error_status = lambda count: COMMON_TEXTS.error_status(count, "jobs")
    submit_failed = lambda prompt_idx, error: f"Failed to submit job for prompt {prompt_idx}: {error}"
    run_selected_prompts_button = lambda count: f"🚀 Run {count} Selected Prompts"
    run_selected_prompts_success = "Successfully submitted jobs"
    run_selected_prompts_error = "Failed to submit jobs"
    show_models_with_correctness = (
        lambda is_correct, models: f"{'✅' if is_correct else '❌'} ({len(models)}): {', '.join(models)}"
    )
    skipping_running = (
        lambda model_arch, model_size: f"Skipping running {model_arch} {model_size} because it is already running"
    )


class INFO_FLOW_TEXTS:
    icon = "📈"
    title = "Info Flow Visualization"
    plot_config_title = "Plot Configuration"
    show_data_sources = "Show Data Sources Tree"
    total_experiments = lambda count: f"Total experiments: {count}"
    generate_plots = "Generate Plots"
    generating_plots = "Generating plots..."
    plots_generated = lambda count: f"Successfully generated {count} plots"
    no_plots_generated = "No plots could be generated. Please check the error details above."


class AppGlobalText:
    window_size = "Window Size"
    gpu_type = "GPU Type"
    code_version = "CodeVersion"


class RESULTS_BANK_TEXTS:
    icon = "📋"
    title = "Results Bank"


class DATA_REQUIREMENTS_TEXTS:
    icon = "📊"
    title = "Data Requirements"
    update_requirements = "🔄 Update Latest Requirements"
    requirements_updated = "Requirements updated successfully!"
    save_overrides = "💾 Save Overrides"
    overrides_saved = "Overrides saved successfully!"


class FINAL_PLOTS_TEXTS:
    icon = "📊"
    title = "Final Plots"

    # Headers
    plot_management = "Plot Management"
    plot_generation = "Plot Generation"
    configuration = "### Configuration"

    # Buttons
    add_plot = "Add New Plot"
    edit_plot = "Edit Plot"
    delete_plot = "Delete Plot"
    generate_plot = "Generate Plot"
    save_plot = "Save Plot"
    run_missing_reqs = "Run Missing Requirements"
    save_all = "Save All"
    reset_to_default = "Reset to Default"
    cancel = "Cancel"

    # Titles
    total_plots_title = "Total Plots"
    lines_per_plot_title = "Lines per Plot"

    # Messages
    no_plots = "No plot plans available. Add a new plot plan to get started."
    plot_saved = lambda path: f"Plot saved to {path}"
    missing_reqs = lambda count: f"Missing data for {count} requirements. Please run the missing requirements first."
    generating_plot = lambda title: f"Generating plot: {title}"
    plot_generated = "Plot generated successfully!"
    plot_generation_failed = "Failed to generate plot. Please check the error details above."
    initializing_plot_plans = "Initializing plot plans with default configurations..."
    default_plot_plans_created = "Default plot plans created successfully!"
    no_plan_selected = "No plot plan selected. Please select a plot plan from the sidebar or create a new one."
    confirm_reset = "Click again to confirm reset to default plot plans."
    plot_plans_saved = "Plot plans saved successfully."
    plot_summary = "### Plot Summary"
    total_plots = lambda count: f"**Total plots:** {count}"
    grid_structure = lambda parts: " × ".join(parts)
    lines_per_plot = lambda count: f"**Lines per plot:** {count}"
    plot_plan_not_found = lambda id: f"Plot plan with ID {id} not found."


class INFO_FLOW_ANALYSIS_TEXTS:
    title = "Info Flow Analysis"
    icon = str("📊")
    no_requirements = "No info flow requirements found in latest fulfilled requirements."
    select_requirement = "Select requirement to analyze"
    accuracy_over_windows = "Accuracy over windows"
    probability_distribution = "Probability Distribution"
    sample_results = "Sample Results"
    sample_results_count = "Sample Results Count"
    seed = "Seed"
    layers_range = "Layers Range"
    TAB_INFO_FLOW_OVER_TIME = "Info Flow Over Time"
    TAB_PROBABILITY_DISTRIBUTION = "Probability Distribution"
    select_metric = "Select metric to plot"
    statistics = "Statistics"
    axis_selection = "Axis Selection"
    col_to_axis_name = lambda axis: ["X", "Y"][axis] + " Axis"
    select_flow = "Select Flow"
    metric_selection = "Metric Selection"
    no_common_indices = "No common indices found in the selected info flow results."
    x_axis = "X Axis"
    y_axis = "Y Axis"


class PROMPTS_COMPARISON_TEXTS:
    title = "Prompts Comparison"
    icon = "🔤"

    class TABS:
        SHOW_UNIQUE_TOKENIZERS = "Show Unique Tokenizers"
        SHOW_METRICS = "Show Metrics"
        SHOW_TOKENIZATION = "Show Tokenization"


class MAMBA_ANALYSIS_TEXTS:
    title = "Mamba Architecture Analysis"
    icon = "🐍"


class PROMPT_FILTERATION_PRESETS_TEXTS:
    title = "Prompt Filteration Presets"
    icon = "🔍"
