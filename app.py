"""Quant Lab Pro — Streamlit entry point.

Run locally with:  streamlit run app.py

This file only sets up the page, the shared sidebar (data, strategy and cost
inputs) and navigation. Every page lives in ``pages/`` and every calculation in
the ``quantlab`` package.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Quant Lab Pro",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on phones, open on desktop
    menu_items={
        "About": "Quant Lab Pro — a research and backtesting workstation for systematic equity "
                 "strategies. Research demonstration only; not investment advice.",
    },
)

from ui import state  # noqa: E402  (must follow set_page_config)
from ui.theme import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

PAGES = [
    st.Page("pages/1_Research_Overview.py", title="Overview", icon=":material/dashboard:", default=True,
            url_path="overview"),
    st.Page("pages/2_Strategy_Lab.py", title="Strategy Lab", icon=":material/science:", url_path="strategy-lab"),
    st.Page("pages/3_Backtesting.py", title="Backtest", icon=":material/receipt_long:", url_path="backtest"),
    st.Page("pages/4_Performance_Analytics.py", title="Analytics", icon=":material/query_stats:",
            url_path="analytics"),
    st.Page("pages/5_Walk_Forward.py", title="Walk-Forward", icon=":material/timeline:", url_path="walk-forward"),
    st.Page("pages/6_Data_Explorer.py", title="Data", icon=":material/database:", url_path="data"),
    st.Page("pages/7_Research_Notes.py", title="Notes", icon=":material/edit_note:", url_path="notes"),
]

navigation = st.navigation(PAGES, position="top")
st.session_state["run_spec"] = state.render_sidebar()
navigation.run()
