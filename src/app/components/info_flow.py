import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame

from src.analysis.plots.info_flow_confidence import (
    TMetricType,
    create_plotly_confidence_chart,
)
from src.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.core.consts import (
    TOKEN_TYPE_COLORS,
    TOKEN_TYPE_LINE_STYLES,
    format_params_for_title,
)
from src.core.names import COLS, InfoFlowMetricName
from src.data_ingestion.data_defs.data_defs import InfoFlowResults
from src.utils.streamlit.helpers.component import StreamlitComponent


class InfoFlowAnalysisComponent(StreamlitComponent):
    def __init__(
        self,
        info_flow_results: InfoFlowResults,
    ):
        self.info_flow_results = info_flow_results
        self.metric_options = {
            "Accuracy": TMetricType.ACC,
            "Probability Difference": TMetricType.DIFF,
        }

    def render_probability_distribution(self):
        """Render info flow over time analysis with confidence intervals using Plotly."""
        if not self.info_flow_results:
            st.warning("No info flow data available")
            return

        # Get common and different parameters
        common_params, different_params_list = self.info_flow_results.get_common_and_different_params()

        # Create title based on common parameters
        title = format_params_for_title(common_params)

        # Create legend labels based on different parameters

        # Add metric selection
        st.subheader("Metric Selection")
        selected_metrics = st.multiselect(
            INFO_FLOW_ANALYSIS_TEXTS.select_metric,
            options=list(self.metric_options.keys()),
            default=list(self.metric_options.keys()),
        )

        if not selected_metrics:
            st.warning("Please select at least one metric to display")
            return

        # Determine number of columns based on selected metrics
        num_cols = len(selected_metrics)

        # Prepare data for confidence plots
        targets_window_outputs = []
        base_probs = []
        colors = []
        line_styles = []
        legend_labels = [format_params_for_title(diff_params) for diff_params in different_params_list]
        if not legend_labels:
            legend_labels = ["Flow"]

        for i, info_flow in enumerate(self.info_flow_results):
            # Create a unique source identifier for each info flow
            targets_window_outputs.append(info_flow.get_outputs())
            base_probs.append(
                info_flow.get_runner_dependencies()["evaluate_model"].get_prompt_data()[
                    COLS.EVALUATE_MODEL.TARGET_PROBS
                ]
            )

            # Assign a custom color
            colors.append(
                TOKEN_TYPE_COLORS.get(
                    info_flow.variant_params.source, px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                )
            )
            line_styles.append(TOKEN_TYPE_LINE_STYLES.get(info_flow.variant_params.feature_category, "-"))

        # Find the maximum layer range across all info flows

        st.subheader(title, divider="rainbow")
        cols = st.columns(num_cols, border=True)
        # Create and display plots for each selected metric
        for i, metric_name in enumerate(selected_metrics):
            metric_type = self.metric_options[metric_name]

            with cols[i]:
                st.subheader(f"{metric_name} Over Time")

                # Create confidence plot using Plotly
                fig = create_plotly_confidence_chart(
                    targets_window_outputs=targets_window_outputs,
                    metric_type=metric_type,
                    confidence_level=0.95,
                    colors=colors,
                    line_styles=line_styles,
                    legend_labels=legend_labels,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Create a combined table with statistics for all metrics
        stats_data = []

        for i, info_flow in enumerate(targets_window_outputs):
            # Extract hit data for accuracy
            hit_data = {}
            for window_idx, window_data in info_flow.items():
                hit_data[window_idx] = window_data["hit"]
            accuracy_df = pd.DataFrame(hit_data).mean()

            # Extract probability data
            true_probs = {}
            for window_idx, window_data in info_flow.items():
                true_probs[window_idx] = window_data[InfoFlowMetricName.true_probs]
            true_probs_df = pd.DataFrame(true_probs)
            base_probs_df = pd.DataFrame(base_probs).reset_index(drop=True)
            prob_diffs = true_probs_df - base_probs_df

            # Calculate statistics
            mean_acc = accuracy_df.mean()
            max_acc = accuracy_df.max()
            min_acc = accuracy_df.min()

            mean_diff = prob_diffs.mean().mean()
            max_diff = prob_diffs.max().max()
            min_diff = prob_diffs.min().min()

            # Add to stats data
            stats_data.append(
                {
                    "Flow": legend_labels[i],
                    "Mean Accuracy": f"{mean_acc:.2%}",
                    "Max Accuracy": f"{max_acc:.2%}",
                    "Min Accuracy": f"{min_acc:.2%}",
                    "Mean Probability Change": f"{mean_diff:.2%}",
                    "Max Probability Change": f"{max_diff:.2%}",
                    "Min Probability Change": f"{min_diff:.2%}",
                }
            )

        # Display the combined statistics table
        st.subheader(INFO_FLOW_ANALYSIS_TEXTS.statistics)
        st.table(pd.DataFrame(stats_data))

    def render_info_flow_over_time(self):
        """Render info flow over time analysis."""
        if not self.info_flow_results:
            st.warning("No info flow data available")
            return

        # Get common and different parameters
        common_params, different_params_list = self.info_flow_results.get_common_and_different_params()

        # Create title based on common parameters
        title = format_params_for_title(common_params)

        # Create legend labels based on different parameters
        legend_labels = [format_params_for_title(diff_params) for diff_params in different_params_list]
        if not legend_labels:
            legend_labels = ["Flow"]

        # Define axis options
        axes_options = {
            "Base Probability": COLS.EVALUATE_MODEL.TARGET_PROBS,
            "True Probability": COLS.INFO_FLOW.TRUE_PROBS,
            "Probability Difference": COLS.INFO_FLOW.DIFFS,
        }

        # Create axis selection
        st.subheader(INFO_FLOW_ANALYSIS_TEXTS.axis_selection)
        cols = st.columns(2)
        selected_info_flow_indices = []
        selected_metrics = []

        for i, col in enumerate(cols):
            with col:
                st.subheader(INFO_FLOW_ANALYSIS_TEXTS.col_to_axis_name(i))
                if self.info_flow_results.size == 1:
                    selected_info_flow_indices.append(0)
                else:
                    selected_info_flow_indices.append(
                        st.selectbox(
                            INFO_FLOW_ANALYSIS_TEXTS.select_flow,
                            options=range(self.info_flow_results.size),
                            key=f"info_flow_output_{i}",
                            format_func=lambda i: legend_labels[i],
                        )
                    )
                selected_metrics.append(
                    st.selectbox(
                        INFO_FLOW_ANALYSIS_TEXTS.metric_selection,
                        list(axes_options.keys()),
                        index=i,
                        key=f"metric_{i}",
                    )
                )

        st.subheader(title, divider="rainbow")
        # Create a figure
        fig = go.Figure()

        # Prepare hover data with additional information from model_evaluations
        hover_columns = [
            COLS.ORIGINAL_IDX,
            COLS.COUNTER_FACT.SUBJECT,
            COLS.COUNTER_FACT.RELATION,
            COLS.COUNTER_FACT.TARGET_TRUE,
            COLS.COUNTER_FACT.TARGET_FALSE,
            COLS.EVALUATE_MODEL.MODEL_OUTPUT,
            COLS.EVALUATE_MODEL.TARGET_RANK,
            COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE,
        ]

        # Get window indices from both selected info flows
        all_window_indices = set()
        info_flows = []

        for idx in selected_info_flow_indices:
            info_flow = self.info_flow_results[idx].get_outputs()
            info_flows.append(info_flow)
            all_window_indices.update(info_flow.keys())

        # Sort window indices
        all_window_indices = sorted(all_window_indices)
        hover_data = [
            "<br>".join([f"<b>{col}:</b> {row[col]}" for col in hover_columns])
            for _, row in self.info_flow_results[selected_info_flow_indices[0]]
            .get_runner_dependencies()["evaluate_model"]
            .get_prompt_data()
            .reset_index()
            .iterrows()
        ]
        if not all_window_indices:
            st.warning("No window indices found in the selected info flows")
            return

        # Prepare data for each axis
        axes_data = []
        for flow_idx, axis_name in zip(selected_info_flow_indices, selected_metrics):
            info_flow = self.info_flow_results[flow_idx].get_outputs()
            axis_column = axes_options[axis_name]

            # Prepare data for this axis
            metric_per_window = {}
            accuracy_by_window = {}
            hit_per_window = {}

            for window_idx in all_window_indices:
                if window_idx not in info_flow:
                    continue

                window_data = info_flow[window_idx]
                accuracy_by_window[window_idx] = np.mean(window_data[InfoFlowMetricName.hit])
                hit_per_window[window_idx] = window_data[InfoFlowMetricName.hit]
                match axis_column:
                    case InfoFlowMetricName.diffs | InfoFlowMetricName.true_probs:
                        values = window_data[axis_column]
                    case COLS.EVALUATE_MODEL.TARGET_PROBS:
                        values: DataFrame = (
                            self.info_flow_results[flow_idx]
                            .get_runner_dependencies()["evaluate_model"]
                            .get_prompt_data()[axis_column]
                        )
                    case _:
                        raise ValueError(f"Unknown axis column: {axis_column}")

                metric_per_window[window_idx] = values

            axes_data.append(
                {
                    "data": pd.DataFrame(metric_per_window),
                    "accuracy_by_window": accuracy_by_window,
                    "flow_name": legend_labels[flow_idx],
                    "hover_data": hover_data,
                    "hit_per_window": hit_per_window,
                }
            )

        # Check if we have data for both axes
        if not all(axes_data):
            st.warning("Missing data for one or both axes")
            pass
            return

        # Create frames for animation
        frames = []
        for window_idx in all_window_indices:
            # Filter data for this window
            x_window_data = np.array(axes_data[0]["data"][window_idx])
            y_window_data = np.array(axes_data[1]["data"][window_idx])

            # Check if lengths match
            if len(x_window_data) != len(y_window_data):
                # Only show warning for the first window with mismatched lengths
                raise ValueError(
                    f"Data length mismatch for window {window_idx}. Some data points may not be displayed correctly."
                )

            marker_colors = []
            for i in range(len(x_window_data)):
                # Get hit/miss information if available
                x_hit = axes_data[0]["hit_per_window"][window_idx][i]
                y_hit = axes_data[1]["hit_per_window"][window_idx][i]

                if x_hit and y_hit:
                    color = "green"  # Both hit
                elif x_hit:
                    color = "blue"  # X hit only
                elif y_hit:
                    color = "orange"  # Y hit only
                else:
                    color = "red"  # Both miss

                marker_colors.append(color)

            # Create frame
            label = (
                f"{window_idx} ({axes_data[0]['accuracy_by_window'][window_idx]:.1%} avg acc)"
                if window_idx in axes_data[0]["accuracy_by_window"]
                else f"{window_idx}"
            )

            frames.append(
                {
                    "name": label,
                    "data": [
                        {
                            "x": x_window_data,
                            "y": y_window_data,
                            "mode": "markers",
                            "marker": {
                                "color": marker_colors,
                                "size": 10,
                            },
                            "text": hover_data,
                            "hoverinfo": "text",
                        }
                    ],
                }
            )

        # Create figure
        fig = go.Figure(
            data=frames[0]["data"],
            layout=go.Layout(
                title=title,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                xaxis=dict(
                    title=f"{axes_data[0]['flow_name']} - {axes_options[selected_metrics[0]]}",
                    zeroline=True,
                    range=[0, 1],  # Fix scale from 0 to 1
                ),
                yaxis=dict(
                    title=f"{axes_data[1]['flow_name']} - {axes_options[selected_metrics[1]]}",
                    zeroline=True,
                    range=[0, 1],  # Fix scale from 0 to 1
                ),
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "y": 0,
                        "x": 0,
                        "xanchor": "right",
                        "yanchor": "top",
                        "pad": dict(t=0, r=10),
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            },
                        ],
                    }
                ],
                sliders=[
                    {
                        "steps": [
                            {
                                "method": "animate",
                                "label": frame["name"],
                                "args": [
                                    [frame["name"]],
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 300},
                                    },
                                ],
                            }
                            for frame in frames
                        ],
                        "active": 0,
                        "currentvalue": {"prefix": "Window: "},
                    }
                ],
            ),
            frames=[
                go.Frame(
                    name=frame["name"],
                    data=frame["data"],
                )
                for frame in frames
            ],
        )

        # Display the figure
        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        # Display analysis
        st.subheader(
            " | ".join(
                [
                    "Analysis",
                    f"{len(self.info_flow_results)} info flows",
                    f"{len(self.info_flow_results[0].input_params.filteration.get_prompt_ids())} prompts",
                ]
            )
        )

        plan = {
            INFO_FLOW_ANALYSIS_TEXTS.TAB_PROBABILITY_DISTRIBUTION: self.render_probability_distribution,
            INFO_FLOW_ANALYSIS_TEXTS.TAB_INFO_FLOW_OVER_TIME: self.render_info_flow_over_time,
        }
        # Tabs for different analysis views
        # tab = st.tabs([tab_name for tab_name in plan])
        tab = [st.expander(tab_name) for tab_name in plan]

        for i, render_func in enumerate(plan.values()):
            with tab[i]:
                with st.spinner("Loading...", show_time=True):
                    render_func()
