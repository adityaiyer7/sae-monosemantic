"""
Cached query functions for the SAE feature explorer.

All expensive DuckDB/HF queries are wrapped with @st.cache_resource or
@st.cache_data so each result is computed at most once per session.

Convention: parameters named with a leading underscore (e.g. _analyzer)
are excluded from Streamlit's hash so unhashable objects can be passed.
"""
import os
import streamlit as st
import pandas as pd

from app.backend.hf_connector import CONFIGS, FEATURES_TABLE, HFFeatureAnalyzer


@st.cache_resource(show_spinner="Connecting to HuggingFace dataset...")
def get_analyzer(config_key: str) -> HFFeatureAnalyzer:
    """
    Return a cached HFFeatureAnalyzer for the given config key.
    One instance is created and reused for the lifetime of the Streamlit process.
    """
    cfg = CONFIGS[config_key]
    hf_token = os.environ["HF_TOKEN"]
    return HFFeatureAnalyzer(
        hf_dataset_path=cfg["hf_path"],
        expansion_factor=cfg["expansion"],
        hf_token=hf_token,
    )


@st.cache_data(show_spinner="Ranking features by selectivity (first load may take a while)...")
def get_feature_rankings(
    config_key: str,
    min_activations: int | None = None,
) -> pd.DataFrame:
    """
    Return the full feature ranking table for a config.
    Cached forever per (config_key, min_activations) pair.
    """
    analyzer = get_analyzer(config_key)
    return analyzer.rank_features_by_selectivity(FEATURES_TABLE, min_activations)


@st.cache_data(show_spinner="Fetching top activations...")
def get_top_activations_with_context(
    config_key: str,
    feature_id: int,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Return top-k activations for a feature, enriched with token text and
    a formatted context_string ready for display.
    """
    analyzer = get_analyzer(config_key)
    df = analyzer.get_top_activations(FEATURES_TABLE, feature_id, top_k)
    if df.empty:
        return df
    df = analyzer.reconstruct_token_text(df)
    df = analyzer.reconstruct_context_text(df)
    df = analyzer.get_context_string(df)
    return df


@st.cache_data(show_spinner="Loading activation distribution...")
def get_activation_stats(config_key: str, feature_id: int) -> dict:
    """Return summary statistics for a feature's activation distribution."""
    analyzer = get_analyzer(config_key)
    return analyzer.get_activation_stats(FEATURES_TABLE, feature_id)


@st.cache_data(show_spinner="Loading activation values...")
def get_activation_values(config_key: str, feature_id: int) -> pd.DataFrame:
    """Return all activation values for a feature (for histogram rendering)."""
    analyzer = get_analyzer(config_key)
    return analyzer.get_activation_values(FEATURES_TABLE, feature_id)


@st.cache_data(show_spinner="Computing co-occurring features...")
def get_co_occurring_features(
    config_key: str,
    feature_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-n features that co-activate with a given feature."""
    analyzer = get_analyzer(config_key)
    df = analyzer.get_co_occuring_features(FEATURES_TABLE, feature_id)
    return df.head(top_n)


@st.cache_data(show_spinner="Counting dead features...")
def get_dead_feature_count(config_key: str) -> int:
    """Return the number of dead (never-firing) features for a config."""
    analyzer = get_analyzer(config_key)
    return len(analyzer.get_dead_features())


@st.cache_data(show_spinner="Running LLM feature analysis...")
def get_llm_label(config_key: str, feature_id: int, top_k: int, use_groq: bool) -> dict:
    """Return LLM-generated label and reasoning for a feature."""
    analyzer = get_analyzer(config_key)
    df = get_top_activations_with_context(config_key, feature_id, top_k)
    if df.empty:
        return {}
    result = analyzer.label_feature(df, use_groq=use_groq)
    row = result.iloc[0]
    return {"label": row["llm_label"], "reasoning": row["llm_reasoning"]}
