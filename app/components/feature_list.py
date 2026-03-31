"""
Feature ranking panel component.

Renders the sortable, selectable table of SAE features ranked by selectivity.
Clicking a row sets st.session_state.selected_feature_id.
"""
import streamlit as st
import pandas as pd

from app.backend.feature_queries import get_feature_rankings


def render_feature_list(config_key: str, min_activations: int | None) -> int | None:
    """
    Render the feature ranking table and return the currently selected feature_id.

    Handles both click-to-select and the jump_to_feature override from the sidebar.
    Returns the selected feature_id (int) or None if nothing is selected yet.
    """
    st.subheader("Feature Rankings")
    st.caption(
        "Features ranked by selectivity = mean_activation / log(num_activations + 1). "
        "Higher = fires strongly on few tokens (more monosemantic)."
    )

    with st.status("Loading features from HuggingFace...", expanded=True) as status:
        rankings: pd.DataFrame = get_feature_rankings(config_key, min_activations)
        status.update(label="Features loaded.", state="complete", expanded=False)

    if rankings.empty:
        st.info("No features found matching the current filter settings.")
        return None

    # Format for display
    display_df = rankings.copy()
    display_df["selectivity_score"] = display_df["selectivity_score"].map("{:.4f}".format)
    display_df["mean_activation"] = display_df["mean_activation"].map("{:.4f}".format)
    display_df = display_df.rename(columns={
        "feature_id": "Feature ID",
        "num_activations": "# Activations",
        "mean_activation": "Mean Activation",
        "unique_token_count": "Unique Tokens",
        "selectivity_score": "Selectivity ↓",
    })

    # Interactive table — single-row selection updates session state
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"feature_table_{config_key}",
    )

    selected_feature_id: int | None = None

    # Check if user clicked a row
    selected_rows = event.selection.get("rows", [])
    if selected_rows:
        row_idx = selected_rows[0]
        selected_feature_id = int(rankings.iloc[row_idx]["feature_id"])
        st.session_state["selected_feature_id"] = selected_feature_id

    # Sidebar jump-to overrides table selection
    if st.session_state.get("jump_to_feature") is not None:
        selected_feature_id = st.session_state.pop("jump_to_feature")
        st.session_state["selected_feature_id"] = selected_feature_id

    # Restore from session state if no new selection
    if selected_feature_id is None:
        selected_feature_id = st.session_state.get("selected_feature_id")

    return selected_feature_id
