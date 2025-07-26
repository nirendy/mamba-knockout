# Purpose: Manage and generate final plots for the report
# High Level Outline:
# 1. Page setup and configuration
# 2. Plot plan management
# 3. Data requirement detection and execution
# 4. Plot generation and export
# Outline Issues:
# - Consider adding batch operations for plot plans
# - Add support for exporting all plots at once
# Outline Compatibility Issues:
# - New file, outline will be implemented

from dataclasses import dataclass

import streamlit as st

from src.analysis.experiment_results.plot_plan import PlotPlan
from src.app.components.data_requirements import RequirementExecution, RequirementsDisplay
from src.app.components.plot_generation import PlotGenerator
from src.app.components.plot_plans import (
    PlotPlanDetailsSummary,
    PlotPlanEditor,
    PlotPlanSelector,
)
from src.app.data_store import load_results_bank
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.types import TPlotID
from src.data_ingestion.data_defs.data_defs import PlotPlans
from src.utils.streamlit.components.aagrid import SelectionMode
from src.utils.streamlit.helpers.allow_nested_expanders import allow_nested_st_elements
from src.utils.streamlit.helpers.cache import CacheWithDependencies
from src.utils.streamlit.helpers.component import StreamlitComponent, StreamlitPage
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase

NEW_LABEL = TPlotID("New")


class _FinalPlotsSessionKeys(SessionKeysBase["_FinalPlotsSessionKeys"]):
    SELECTED_PLOT_PLAN_ID = SessionKeyDescriptor[TPlotID](TPlotID(NEW_LABEL))
    EDIT_MODE_KEY = SessionKeyDescriptor[bool](False)
    CONFIRM_RESET = SessionKeyDescriptor[bool](False)

    def is_new_plot_plan(self) -> bool:
        return self.SELECTED_PLOT_PLAN_ID.value == NEW_LABEL

    def display_plot_plan_name(self) -> str:
        return "NEW" if self.is_new_plot_plan() else self.SELECTED_PLOT_PLAN_ID.value


FinalPlotsSessionKeys = _FinalPlotsSessionKeys()


def _plot_plan_hash_func(plot_plan: PlotPlan):
    return plot_plan.plot_id


@CacheWithDependencies(max_entries=1, hash_funcs={PlotPlan: _plot_plan_hash_func})
def _load_plot_plan_fulfilled_reqs(plot_plan: PlotPlan):
    result_bank = load_results_bank()

    return plot_plan.get_data_requirements(result_bank).to_fulfilled_reqs(result_bank).summarize()


def save_plot_plans(plot_plans: PlotPlans) -> None:
    """Save plot plans to file."""
    plot_plans.save()
    FinalPlotsSessionKeys.EDIT_MODE_KEY.post_external_update(False)
    st.success(FINAL_PLOTS_TEXTS.plot_plans_saved)


class ManagePlotPlans(StreamlitComponent[None]):
    def __init__(self, plot_plans: PlotPlans):
        self.plot_plans = plot_plans

    def render(self):
        # Buttons for adding/editing/deleting plot plans
        col1, col2 = st.columns(2)

        selected_plot_changed = FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.is_changed

        # Plot plan selector
        with st.expander(
            FinalPlotsSessionKeys.display_plot_plan_name(),
            expanded=False,
        ):
            PlotPlanSelector(
                plot_plans=self.plot_plans,
                selected_plot_id_sk=FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID,
                new_label=NEW_LABEL,
            ).render()

        if selected_plot_changed:
            FinalPlotsSessionKeys.EDIT_MODE_KEY.value = False

        with col1:
            st.checkbox(
                FINAL_PLOTS_TEXTS.edit_plot,
                key=FinalPlotsSessionKeys.EDIT_MODE_KEY.key_for_component,
                disabled=FinalPlotsSessionKeys.is_new_plot_plan(),
            )

        with col2:
            if st.button(
                FINAL_PLOTS_TEXTS.delete_plot,
                use_container_width=True,
                disabled=FinalPlotsSessionKeys.is_new_plot_plan(),
            ):
                self.plot_plans.remove_plan(FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value)
                save_plot_plans(self.plot_plans)
                FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.default_value = NEW_LABEL
                st.rerun()


@dataclass
class PlotPlanRequirements(StreamlitComponent[None]):
    """Component for displaying and managing data requirements for a plot plan."""

    plot_plan: PlotPlan

    def render(self):
        # Get data requirements for the plot plan
        with st.sidebar:
            fulfilled_reqs = _load_plot_plan_fulfilled_reqs.call_and_render(self.plot_plan)

        with st.expander(
            f"Data Requirements (Missing: {fulfilled_reqs.amount_missing()})",
            expanded=fulfilled_reqs.amount_missing() > 0,
        ):
            data_reqs_to_run = RequirementsDisplay(
                fulfilled_reqs,
                height=400,
                selection_mode=SelectionMode.MULTIPLE,
                hide_columns=[],
                key=f"plot_plan_requirements_{self.plot_plan.title}",
            ).render()

            # Option to run missing requirements
            if data_reqs_to_run is not None:
                with allow_nested_st_elements():
                    RequirementExecution(data_reqs_to_run).render()


class FinalPlotsPage(StreamlitPage):
    def render(self):
        plot_plans: PlotPlans = PlotPlans.load()
        with st.sidebar:
            st.subheader(FINAL_PLOTS_TEXTS.plot_management)
            ManagePlotPlans(plot_plans).render()
            result_bank = load_results_bank()

        # Main area
        if FinalPlotsSessionKeys.EDIT_MODE_KEY.value or FinalPlotsSessionKeys.is_new_plot_plan():
            # Edit mode
            plot_plan: PlotPlan | None = PlotPlanEditor(
                plot_plans,
                None if FinalPlotsSessionKeys.is_new_plot_plan() else FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value,
            ).render()

            if st.button(FINAL_PLOTS_TEXTS.cancel):
                FinalPlotsSessionKeys.EDIT_MODE_KEY.post_external_update(False)

            if plot_plan:
                if FinalPlotsSessionKeys.is_new_plot_plan():
                    # Add new plan
                    plot_plans = plot_plans.add_plan(plot_plan)

                save_plot_plans(plot_plans)

        elif FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value:
            # Display mode
            selected_plan = plot_plans.get_plan(FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value)
            assert selected_plan is not None
            # Display plan details
            with st.expander("Details"):
                PlotPlanDetailsSummary(plot_plans, FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value).render()

            # Display data requirements
            PlotPlanRequirements(selected_plan).render()

            # Plot generation button
            plot_path = PlotGenerator(selected_plan, result_bank).render()
            if plot_path:
                st.toast(FINAL_PLOTS_TEXTS.plot_saved(plot_path))

        else:
            # No plan selected
            st.info(FINAL_PLOTS_TEXTS.no_plan_selected)


if __name__ == "__main__":
    st.set_page_config(page_title=FINAL_PLOTS_TEXTS.title, page_icon=FINAL_PLOTS_TEXTS.icon, layout="wide")
    st.title(f"{FINAL_PLOTS_TEXTS.title} {FINAL_PLOTS_TEXTS.icon}")

    FinalPlotsPage().render()
