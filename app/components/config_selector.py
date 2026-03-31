"""
Sidebar config selector component.

Renders the configuration picker and navigation controls in the Streamlit
sidebar, then returns the user's selections as a dict.
"""
import os
import streamlit as st
from app.backend.hf_connector import CONFIGS


def render_config_selector() -> dict:
    """
    Render the sidebar and return the current UI selections.

    Returns a dict with:
        config_key      (str)       currently selected config label
        top_k           (int)       number of top activations to fetch per feature
        min_activations (int|None)  minimum activation count filter for feature list
        jump_to_feature (int|None)  feature index to jump to directly, or None
    """
    with st.sidebar:
        st.title("SAE Feature Explorer")
        st.caption("GPT-2 Residual Stream · Sparse Autoencoders")
        st.divider()

        config_key = st.selectbox(
            "Model config",
            options=list(CONFIGS.keys()),
            help="Select the SAE expansion factor, sparsity penalty (λ), and whether "
                 "activation-weighted sampling was used during training.",
        )

        cfg = CONFIGS[config_key]
        st.caption(
            f"{cfg['num_features']:,} features · "
            f"GPT-2 layer 6 residual stream"
        )

        st.divider()
        st.subheader("Feature list options")

        min_activations = st.slider(
            "Min activations filter",
            min_value=0,
            max_value=100,
            value=5,
            help="Hide features with fewer than this many stored activation records. "
                 "Useful for filtering near-dead features.",
        )

        st.divider()
        st.subheader("Feature detail options")

        top_k = st.slider(
            "Top activations to show",
            min_value=5,
            max_value=25,
            value=15,
            help="Number of highest-activation examples to display for the selected feature.",
        )

        st.divider()
        st.subheader("Jump to feature")

        jump_input = st.text_input(
            "Feature index",
            value="",
            placeholder=f"0 – {cfg['num_features'] - 1}",
            help="Enter a feature index to navigate directly to it.",
        )
        jump_to_feature = None
        if jump_input.strip().isdigit():
            idx = int(jump_input.strip())
            if 0 <= idx < cfg["num_features"]:
                jump_to_feature = idx
            else:
                st.warning(
                    f"Index must be between 0 and {cfg['num_features'] - 1}."
                )

        st.divider()

        # LLM analysis — only show providers whose API key is present
        has_groq = bool(os.environ.get("GROQ_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        available_providers = []
        if has_groq:
            available_providers.append("Groq (openai/gpt-oss-120b)")
        if has_openai:
            available_providers.append("OpenAI (gpt-5.4)")

        llm_provider = None
        if available_providers:
            st.subheader("LLM Analysis")
            llm_provider = st.selectbox(
                "Provider",
                options=available_providers,
                help="Select which LLM to use for automatic feature labelling. "
                     "Only providers with a configured API key are shown.",
            )
        else:
            st.caption("Set GROQ_API_KEY or OPENAI_API_KEY in .env to enable LLM feature labelling.")

        st.divider()
        st.caption(
            "Data served from HuggingFace · "
            "First load per config may take 30–60 s while DuckDB scans parquets."
        )

    return {
        "config_key": config_key,
        "top_k": top_k,
        "min_activations": min_activations if min_activations > 0 else None,
        "jump_to_feature": jump_to_feature,
        "llm_provider": llm_provider,
    }
