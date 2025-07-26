import streamlit as st

# from scripts.result_bank_migration_scripts.prompt_id_to_h5 import MigrateResults
from src.app.components.result_bank import ShowResultsBank
from src.app.data_store import load_results_bank, load_test_results_bank
from src.app.texts import RESULTS_BANK_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage


class ResultsBankPage(StreamlitPage):
    def render(self):
        with st.sidebar:
            is_test_results = st.checkbox("Show Test Results")
            # is_migrate_results = st.checkbox("Migrate Results")

        result_bank_func = load_test_results_bank if is_test_results else load_results_bank

        results_bank = result_bank_func()
        result_bank_func.render()
        ShowResultsBank(results_bank, key=f"results_bank_{is_test_results}").render()

        # if is_migrate_results:
        #     MigrateResults(results_bank, is_test_results).render()


if __name__ == "__main__":
    st.set_page_config(page_title=RESULTS_BANK_TEXTS.title, page_icon=RESULTS_BANK_TEXTS.icon, layout="wide")
    st.title(f"{RESULTS_BANK_TEXTS.title} {RESULTS_BANK_TEXTS.icon}")

    ResultsBankPage().render()
