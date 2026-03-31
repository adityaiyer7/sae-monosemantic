"""
SAE Feature Explorer — main Streamlit entry point.

Run locally:
    streamlit run app/streamlit_app.py

Run via Docker:
    docker compose up
"""
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` and `src.*` imports work
# whether launched from the repo root or from inside the app/ directory.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# --- HF_TOKEN guard --------------------------------------------------------
# Must be set before importing anything that touches HuggingFace.
if not os.environ.get("HF_TOKEN"):
    st.set_page_config(page_title="SAE Feature Explorer", layout="wide")
    st.error(
        "**HF_TOKEN is not set.**\n\n"
        "This app queries HuggingFace datasets and requires an access token.\n\n"
        "**How to fix:**\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Add your HuggingFace token: `HF_TOKEN=hf_...`\n"
        "3. Restart with `docker compose up` (or `streamlit run app/streamlit_app.py`)"
    )
    st.stop()

from app.components.config_selector import render_config_selector
from app.components.feature_list import render_feature_list
from app.components.feature_view import render_feature_view

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SAE Feature Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "selected_feature_id" not in st.session_state:
    st.session_state["selected_feature_id"] = None
if "active_config_key" not in st.session_state:
    st.session_state["active_config_key"] = None

# ---------------------------------------------------------------------------
# Sidebar — config + navigation controls
# ---------------------------------------------------------------------------

selections = render_config_selector()
config_key = selections["config_key"]
top_k = selections["top_k"]
min_activations = selections["min_activations"]
llm_provider = selections["llm_provider"]

# If config changed, clear the selected feature so stale state isn't shown
if st.session_state["active_config_key"] != config_key:
    st.session_state["selected_feature_id"] = None
    st.session_state["active_config_key"] = config_key

# Pass sidebar jump request into session state for feature_list to consume
if selections["jump_to_feature"] is not None:
    st.session_state["jump_to_feature"] = selections["jump_to_feature"]

# ---------------------------------------------------------------------------
# Main layout: feature list on left, detail view on right
# ---------------------------------------------------------------------------

list_col, detail_col = st.columns([2, 3], gap="large")

with list_col:
    selected_feature_id = render_feature_list(config_key, min_activations)

with detail_col:
    if selected_feature_id is None:
        st.info(
            "Select a feature from the table on the left to explore it, "
            "or use **Jump to feature** in the sidebar."
        )
    else:
        render_feature_view(config_key, selected_feature_id, top_k, llm_provider)
