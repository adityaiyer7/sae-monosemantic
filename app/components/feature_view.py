"""
Feature detail panel component.

Renders the full analysis view for a single SAE feature:
  - Stats card (mean, std, unique tokens, activation count)
  - Top activating examples with token highlighting
  - Activation value histogram (Plotly)
  - Co-occurring features table
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.backend.feature_queries import (
    get_top_activations_with_context,
    get_activation_stats,
    get_activation_values,
    get_co_occurring_features,
    get_llm_label,
)


def render_feature_view(config_key: str, feature_id: int, top_k: int, llm_provider: str | None) -> None:
    """Render the full detail panel for the given feature_id."""
    st.subheader(f"Feature {feature_id}")
    st.divider()

    _render_stats_card(config_key, feature_id)
    st.divider()
    if llm_provider is not None:
        _render_llm_analysis(config_key, feature_id, top_k, llm_provider)
        st.divider()
    _render_top_activations(config_key, feature_id, top_k)
    st.divider()
    _render_histogram(config_key, feature_id)
    st.divider()
    _render_co_occurring(config_key, feature_id)


# ---------------------------------------------------------------------------
# Sub-panels
# ---------------------------------------------------------------------------

def _render_stats_card(config_key: str, feature_id: int) -> None:
    stats = get_activation_stats(config_key, feature_id)
    if not stats:
        st.warning("No activation data found for this feature.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Activation count", f"{stats['count']:,}")
    col2.metric("Mean activation", f"{stats['mean']:.4f}")
    col3.metric("Std deviation", f"{stats['std']:.4f}")
    col4.metric("Unique tokens", f"{stats['unique_tokens']:,}")

    col5, col6 = st.columns(2)
    col5.metric("25th pct", f"{stats['p25']:.4f}")
    col6.metric("75th pct", f"{stats['p75']:.4f}")


def _render_top_activations(config_key: str, feature_id: int, top_k: int) -> None:
    st.markdown("#### Top Activating Examples")
    st.caption(
        "The **bold** token is the one that triggered the feature. "
        "Surrounding tokens show ±10 context window."
    )

    df: pd.DataFrame = get_top_activations_with_context(config_key, feature_id, top_k)

    if df.empty:
        st.info("No activations found for this feature.")
        return

    for _, row in df.iterrows():
        activation_val = row["activation_value"]
        context = row.get("context_string", "")

        with st.container(border=True):
            # Activation badge + context on same row
            badge_col, text_col = st.columns([1, 8])
            badge_col.metric(
                label="activation",
                value=f"{activation_val:.3f}",
                label_visibility="collapsed",
            )
            # Render context_string as markdown so **bold** highlights the token
            text_col.markdown(
                f"`{activation_val:.3f}` &nbsp; {context}",
                unsafe_allow_html=False,
            )


def _render_histogram(config_key: str, feature_id: int) -> None:
    st.markdown("#### Activation Distribution")

    vals_df: pd.DataFrame = get_activation_values(config_key, feature_id)
    if vals_df.empty:
        st.info("No activation data to plot.")
        return

    vals = vals_df["activation_value"].tolist()
    stats = get_activation_stats(config_key, feature_id)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=50,
        marker_color="steelblue",
        opacity=0.8,
        name="Activations",
    ))

    # Mean and median lines
    fig.add_vline(
        x=stats["mean"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean {stats['mean']:.3f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=stats["median"],
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Median {stats['median']:.3f}",
        annotation_position="top left",
    )
    # IQR shading
    fig.add_vrect(
        x0=stats["p25"],
        x1=stats["p75"],
        fillcolor="rgba(0,200,100,0.08)",
        line_width=0,
        annotation_text="IQR",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Activation Value",
        yaxis_title="Count",
        showlegend=False,
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")

    st.plotly_chart(fig, use_container_width=True)


def _render_llm_analysis(config_key: str, feature_id: int, top_k: int, llm_provider: str) -> None:
    use_groq = llm_provider.startswith("Groq")
    result = get_llm_label(config_key, feature_id, top_k, use_groq)
    st.markdown("#### LLM Feature Analysis")
    if not result:
        st.info("No activations to analyse.")
        return
    st.markdown(f"**Label:** {result['label']}")
    if result.get("reasoning"):
        with st.expander("Reasoning"):
            st.markdown(result["reasoning"])


def _render_co_occurring(config_key: str, feature_id: int) -> None:
    st.markdown("#### Co-occurring Features")
    st.caption(
        "Features that fire on the same tokens as this one (top-quartile activations). "
        "High co-occurrence suggests shared semantic territory."
    )

    df: pd.DataFrame = get_co_occurring_features(config_key, feature_id, top_n=10)

    if df.empty:
        st.info("No co-occurring features found.")
        return

    display = df[["feature_id", "co_occurrence_count"]].copy()
    display.columns = ["Feature ID", "Shared Token Count"]
    st.dataframe(display, use_container_width=True, hide_index=True)
