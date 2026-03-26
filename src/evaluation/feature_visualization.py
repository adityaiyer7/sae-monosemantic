import os
import json
import warnings
import duckdb
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub import login
from huggingface_hub import whoami
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import spacy
from openai import OpenAI
from src.evaluation.prompts import LLM_JUDGE_CLASSIFICATION_SYSTEM, LLM_JUDGE_CLASSIFICATION_USER


# Note: feature_id is the SAE feature dimension index


user = whoami(token=os.getenv("HF_TOKEN"))


class FeatureAnalyzer:
    def __init__(
        self,
        HF_dataset_path: str,
        db_name: str,
        expansion_factor: int,
        model_hidden_dim_size: int = 768,
        context_window: int = 10,
        groq_model: str = "openai/gpt-oss-120b",
        openai_model: str = "gpt-5.4",
    ) -> None:
        """
        Initialize the FeatureAnalyzer with a HuggingFace dataset and a local DuckDB database.

        Args:
            HF_dataset_path: HuggingFace dataset repo path (e.g. "user/SAE_monosemanticity_features_32x_0.0001").
            db_name: Base name for the local DuckDB database file (without the .db extension).
            expansion_factor: SAE expansion factor (number of features = expansion_factor * model_hidden_dim_size).
            model_hidden_dim_size: Hidden dimension size of the base transformer model. Defaults to 768 (GPT-2).
            context_window: Number of surrounding tokens on each side of the activating token. Defaults to 10.
            groq_model: Model identifier used when routing LLM calls through Groq. Defaults to "openai/gpt-oss-120b".
            openai_model: Model identifier used when calling OpenAI directly. Defaults to "gpt-5.4-mini".
        """
        self.hf_dataset_path = HF_dataset_path
        self.db_name = db_name
        self.con = duckdb.connect(f'{self.db_name}.db')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.con.execute("DROP SECRET IF EXISTS hf_token")

        self.con.execute(f"""
        CREATE SECRET hf_token (TYPE huggingface, TOKEN '{os.getenv("HF_TOKEN")}')
        """)
        self.expansion_factor = expansion_factor
        self.model_hidden_dim_size = model_hidden_dim_size
        self.context_window = context_window

        self.groq_model = os.getenv("LLM_MODEL", groq_model)
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.openai_model = os.getenv("LLM_MODEL", openai_model)

        self.build_vocab_table()

    # -------------------------------------------------------------------------
    # Token and Context Reconstruction Methods
    # -------------------------------------------------------------------------

    def reconstruct_token_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join each row's token_id to its human-readable token string via the vocab table.

        Args:
            df: DataFrame containing at least a ``token_id`` column.

        Returns:
            A copy of ``df`` with an additional ``token_text`` column populated from
            the vocab lookup table.
        """
        self.con.register("_input", df)
        result = self.con.execute("""
            SELECT s.*, v.token_text
            FROM _input s
            JOIN vocab v ON s.token_id = v.token_id
        """).df()
        self.con.unregister("_input")
        return result

    def reconstruct_context_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode the list of context token IDs in each row into a list of token strings.

        Args:
            df: DataFrame containing at least a ``context_token_ids`` column (array of int).

        Returns:
            A copy of ``df`` with an additional ``context_text`` column, where each entry
            is an ordered list of token strings corresponding to ``context_token_ids``.
        """
        RECONSTRUCT_CONTEXT_QUERY = """
            SELECT s.*, (
                SELECT list(v.token_text ORDER BY pos)
                FROM unnest(s.context_token_ids) WITH ORDINALITY AS u(tid, pos)
                JOIN vocab v ON u.tid = v.token_id
            ) AS context_text
            FROM _input s
        """
        self.con.register("_input", df)
        result = self.con.execute(RECONSTRUCT_CONTEXT_QUERY).df()
        self.con.unregister("_input")
        return result

    def get_context_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a single readable string per row by joining context tokens, with the
        activating token wrapped in ``**bold**`` markers.

        Requires ``reconstruct_context_text`` and ``reconstruct_token_text`` to have
        been called on ``df`` first so that ``context_text`` and ``token_text`` columns
        are present.

        Args:
            df: DataFrame with ``context_text`` (list[str]) and ``token_text`` (str) columns.

        Returns:
            The same DataFrame with an added ``context_string`` column containing the
            formatted context string for each row.
        """
        def build_string(row):
            context_list = []
            for idx, text in enumerate(row["context_text"]):
                if idx == self.context_window:
                    context_list.append(f"**{row['token_text']}**")
                else:
                    context_list.append(text)
            return ''.join(context_list)

        df["context_string"] = df.apply(build_string, axis=1)
        return df

    # -------------------------------------------------------------------------
    # Per Feature Analysis
    # -------------------------------------------------------------------------

    def get_top_activations(
        self,
        table_name: str,
        feature_id: int,
        top_k: int,
        sort_order: str = 'descending',
    ) -> pd.DataFrame:
        """
        Return the top-k activation records for a given feature, ranked by activation value.

        Args:
            table_name: Name of the activations table to query.
            feature_id: SAE feature dimension index to filter on.
            top_k: Number of records to return.
            sort_order: ``"descending"`` (highest activations first) or ``"ascending"``
                (lowest activations first). Defaults to ``"descending"``.

        Returns:
            DataFrame of up to ``top_k`` rows from ``table_name`` where
            ``feature_id`` matches, ordered by ``activation_value``.

        Raises:
            ValueError: If ``sort_order`` is not ``"ascending"`` or ``"descending"``.
        """
        sort_order_map = {'descending': 'DESC', 'ascending': 'ASC'}
        if sort_order not in ('ascending', 'descending'):
            raise ValueError("sort_order must be either descending or ascending")
        sort_order_sql = sort_order_map[sort_order]
        TOP_K_ACTIVATIONS_QUERY = f"""
        SELECT *
        FROM {table_name}
        WHERE feature_id = {feature_id}
        ORDER BY activation_value {sort_order_sql}
        LIMIT {top_k}
        """
        return self.con.execute(TOP_K_ACTIVATIONS_QUERY).df()

    def get_most_active_features(self, table_name: str) -> pd.DataFrame:
        """
        Count how many distinct activation records exist per feature dimension.
        Note that this is not in terms of activation value. 

        Args:
            table_name: Name of the activations table to query.

        Returns:
            DataFrame with columns ``[feature_id, num_features]`` sorted descending
            by ``num_features`` (number of activation records for that feature).
        """
        NUM_FEATURES_QUERY = f"""
            SELECT feature_id, COUNT(*) AS num_features
            FROM {table_name}
            GROUP BY feature_id
            ORDER BY num_features DESC
        """
        return self.con.execute(NUM_FEATURES_QUERY).df()

    def rank_features_by_selectivity(self, table_name: str, min_activations: int | None = None) -> pd.DataFrame:
        """
        Rank all active features by a selectivity score.

        Selectivity is defined as ``mean_activation / log(num_activations + 1)``, which
        rewards features that fire with high magnitude but infrequently — a useful proxy
        for monosemanticity. Dead features (zero activations) are excluded automatically
        since they produce no rows in the activations table.

        Args:
            table_name: Name of the activations table to query.
            min_activations: If provided, exclude features with fewer than this many
                activation records. Useful for filtering out features that fire too
                rarely to be meaningfully interpreted. Defaults to ``None`` (no filter).

        Returns:
            DataFrame with columns ``[feature_id, num_activations, mean_activation,
            unique_token_count, selectivity_score]``, sorted descending by
            ``selectivity_score``. Use ``.head(k)`` for the most selective features
            and ``.tail(k)`` for the least selective.
        """
        having_clause = f"HAVING COUNT(*) >= {min_activations}" if min_activations is not None else ""
        SELECTIVITY_QUERY = f"""
            SELECT
                feature_id,
                COUNT(*) AS num_activations,
                AVG(activation_value) AS mean_activation,
                COUNT(DISTINCT token_id) AS unique_token_count,
                AVG(activation_value) / LOG(COUNT(*) + 1) AS selectivity_score
            FROM {table_name}
            GROUP BY feature_id
            {having_clause}
            ORDER BY selectivity_score DESC
        """
        return self.con.execute(SELECTIVITY_QUERY).df()

    def get_feature_density(self, table_name: str, num_unique_tokens: int = 50257) -> pd.DataFrame:
        """
        Compute the activation density of each feature as a fraction of total activation events.

        Density is defined as: ``COUNT(activations for this feature) / COUNT(all activations)``.
        This answers: "out of all token activations above threshold, what fraction fired this feature?"

        Args:
            table_name: Name of the activations table to query.
            num_unique_tokens: Vocabulary size of the tokenizer. Defaults to 50257 (GPT-2).
                Currently unused in the query but reserved for future normalisation variants.

        Returns:
            DataFrame with columns ``[feature_id, feature_density]``.
        """
        FEATURE_DENSITY_QUERY = f"""
            SELECT feature_id, COUNT(*)/(SELECT COUNT(*) FROM {table_name}) AS feature_density
            FROM {table_name}
            GROUP BY feature_id
        """
        return self.con.execute(FEATURE_DENSITY_QUERY).df()

    def get_activation_distribution(
        self,
        table_name: str,
        feature_id: int,
        save_figs: bool = False,
        figs_dir: str = "figs",
    ) -> dict[str, float | int]:
        """
        Compute summary statistics for the activation value distribution of a single feature.

        Optionally saves a histogram of the distribution annotated with mean, median,
        and interquartile range to disk.

        Args:
            table_name: Name of the activations table to query.
            feature_id: SAE feature dimension index to analyse.
            save_figs: If ``True``, write a PNG histogram to ``figs_dir``. Defaults to ``False``.
            figs_dir: Directory where figures are saved. Created if it does not exist.
                Defaults to ``"figs"``.

        Returns:
            Dictionary with the following keys:

            - ``mean_activation_score`` (float): Mean activation value.
            - ``median_activation_score`` (float): Median activation value.
            - ``standard_deviatin_activation_scores`` (float): Standard deviation.
            - ``activation_value_25th_percentile`` (float): 25th-percentile activation.
            - ``activation_value_75th_percentile`` (float): 75th-percentile activation.
            - ``unique_token_id_count`` (int): Number of distinct token IDs that fired this feature.
        """
        FILTER_QUERY = f"""
        SELECT *
        FROM {table_name}
        WHERE feature_id = {feature_id}
        """

        filtered_df = self.con.execute(FILTER_QUERY).df()
        mean_activation_score = filtered_df["activation_value"].mean()
        median_activation_score = filtered_df["activation_value"].median()
        standard_deviatin_activation_scores = filtered_df["activation_value"].std()
        activation_value_25th_percentile = filtered_df["activation_value"].quantile(q=0.25)
        activation_value_75th_percentile = filtered_df["activation_value"].quantile(q=0.75)

        unique_token_ids = set(filtered_df["token_id"])
        unique_token_id_count = len(unique_token_ids)

        if save_figs:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(filtered_df["activation_value"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(mean_activation_score, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_activation_score:.3f}")
            ax.axvline(median_activation_score, color="orange", linestyle="--", linewidth=1.5, label=f"Median: {median_activation_score:.3f}")
            ax.axvline(activation_value_25th_percentile, color="green", linestyle=":", linewidth=1.2, label=f"25th pct: {activation_value_25th_percentile:.3f}")
            ax.axvline(activation_value_75th_percentile, color="purple", linestyle=":", linewidth=1.2, label=f"75th pct: {activation_value_75th_percentile:.3f}")
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Count")
            ax.set_title(f"Activation Distribution — Feature {feature_id}")
            ax.legend()
            fig.tight_layout()
            Path(figs_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{figs_dir}/feature_{feature_id}_activation_dist.png", dpi=150)
            plt.close(fig)

        return {
            "mean_activation_score": mean_activation_score,
            "median_activation_score": median_activation_score,
            "standard_deviatin_activation_scores": standard_deviatin_activation_scores,
            "activation_value_25th_percentile": activation_value_25th_percentile,
            "activation_value_75th_percentile": activation_value_75th_percentile,
            "unique_token_id_count": unique_token_id_count,
        }

    def get_activation_distribution_per_token_id(
        self,
        table_name: str,
        token_id: int,
    ) -> dict[str, set[int] | int]:
        """
        For a given token ID, find all SAE features that activate for it (across all contexts).

        Args:
            table_name: Name of the activations table to query.
            token_id: The vocabulary token ID to analyse.

        Returns:
            Dictionary with the following keys:

            - ``features_activated_by_token`` (set[int]): Set of feature IDs that fire for this token.
            - ``num_features_activated_by_token`` (int): Cardinality of the above set.
        """
        FILTER_QUERY = f"""
        SELECT *
        FROM {table_name}
        WHERE token_id = {token_id}
        """
        filtered_df = self.con.execute(FILTER_QUERY).df()
        features_activated_by_token = set(filtered_df["feature_id"])
        num_features_activated_by_token = len(features_activated_by_token)

        return {
            "features_activated_by_token": features_activated_by_token,
            "num_features_activated_by_token": num_features_activated_by_token,
        }

    def get_co_occuring_features(
        self,
        table_name: str,
        feature_id: int,
        activation_percentile: float = 0.75,
    ) -> pd.DataFrame:
        """
        For a given feature, find which other features frequently fire on the same tokens.

        Only tokens whose activation value for feature_id meets or exceeds the specified
        percentile threshold are considered — this avoids polluting results with tokens
        that only marginally activate the feature.

        A percentile threshold is used instead of a fixed value to make the result
        scale-invariant across features with different activation magnitudes.

        Args:
            table_name: Name of the activations table.
            feature_id: The SAE feature dimension index to analyse.
            activation_percentile: Percentile (0–1) used to threshold activations for
                ``feature_id`` before computing co-occurrences. Defaults to 0.75 (top quartile).

        Returns:
            DataFrame with columns ``[feature_id, co_occurrence_count, token_ids]``, sorted
            descending by ``co_occurrence_count`` (number of distinct tokens shared with
            ``feature_id``).

        Raises:
            ValueError: If ``activation_percentile`` is outside [0, 1].
        """
        if not (0 <= activation_percentile <= 1):
            raise ValueError("activation_percentile must be between 0 and 1")

        THRESHOLD_QUERY = f"""
        SELECT QUANTILE_CONT(activation_value, {activation_percentile}) AS threshold
        FROM {table_name}
        WHERE feature_id = {feature_id}
        """
        threshold = self.con.execute(THRESHOLD_QUERY).fetchone()[0]

        CO_OCCURING_QUERY = f"""
        SELECT
            t.feature_id,
            COUNT(DISTINCT t.token_id) AS co_occurrence_count,
            LIST(DISTINCT t.token_id) AS token_ids
        FROM {table_name} t
        WHERE t.token_id IN (
            SELECT token_id
            FROM {table_name}
            WHERE feature_id = {feature_id}
            AND activation_value >= {threshold}
        )
        AND t.feature_id != {feature_id}
        GROUP BY t.feature_id
        ORDER BY co_occurrence_count DESC
        """
        return self.con.execute(CO_OCCURING_QUERY).df()

    # -------------------------------------------------------------------------
    # Cross Feature Analysis
    # -------------------------------------------------------------------------

    def get_dead_features(self) -> pd.DataFrame:
        """
        Return features that never fired for any input token during training.

        Dead feature IDs are derived by computing the complement of the alive features
        set (loaded from a JSON file on HuggingFace) against the full feature index range
        ``[0, expansion_factor * model_hidden_dim_size)``.

        Returns:
            DataFrame with a single column ``feature_id`` listing all dead feature indices
            in ascending order.
        """
        total_num_features = self.expansion_factor * self.model_hidden_dim_size

        # Derive the alive_features filename from the HF dataset path
        # e.g. "thedarkknight7/SAE_monosemanticity_features_32x_0.0001" -> "alive_features_32x_0.0001.json"
        repo_name = self.hf_dataset_path.split("/")[-1]
        suffix = repo_name.replace("SAE_monosemanticity_features_", "")
        alive_features_filename = f"alive_features_{suffix}.json"

        local_path = hf_hub_download(
            repo_id=self.hf_dataset_path,
            filename=alive_features_filename,
            repo_type="dataset",
        )
        with open(local_path, "r") as f:
            alive_features = set(json.load(f))

        all_features = set(range(total_num_features))
        dead_feature_ids = sorted(all_features - alive_features)

        return pd.DataFrame({"feature_id": dead_feature_ids})

    def feature_similarity_cosine_similarity(
        self,
        table_name: str,
        feature_id_i: int,
        feature_id_j: int,
    ) -> float:
        """
        Compute cosine similarity between two SAE features based on their activation patterns
        over the token vocabulary.

        Each feature is represented as a sparse vector over token_id space, where each entry
        is the max activation value observed for that token across all contexts. Cosine similarity
        between these vectors captures whether the two features tend to fire on the same tokens
        with similar magnitudes.

        Note: this operates in token-type space (aggregated by token_id), not context space.
        Two features that fire on the same token in different contexts will appear similar even
        if they represent distinct concepts. See TODO.md for a discussion of context-aware similarity.

        Args:
            table_name: Name of the activations table.
            feature_id_i: First SAE feature dimension index.
            feature_id_j: Second SAE feature dimension index.

        Returns:
            Cosine similarity in [-1, 1]. Returns 0.0 if either feature has no activations.
        """
        COS_SIM_QUERY = f"""
        WITH agg AS (
            SELECT feature_id, token_id, MAX(activation_value) AS activation
            FROM {table_name}
            WHERE feature_id IN ({feature_id_i}, {feature_id_j})
            GROUP BY feature_id, token_id
        ),
        norms AS (
            SELECT feature_id, SQRT(SUM(activation * activation)) AS norm
            FROM agg
            GROUP BY feature_id
        )
        SELECT
            SUM(a.activation * b.activation) / (ANY_VALUE(na.norm) * ANY_VALUE(nb.norm)) AS cosine_similarity
        FROM agg a
        JOIN agg b ON a.token_id = b.token_id AND a.feature_id = {feature_id_i} AND b.feature_id = {feature_id_j}
        JOIN norms na ON na.feature_id = {feature_id_i}
        JOIN norms nb ON nb.feature_id = {feature_id_j}
        """
        result = self.con.execute(COS_SIM_QUERY).fetchone()
        if result is None or result[0] is None:
            return 0.0
        return result[0]

    def feature_similarity_correlation(
        self,
        table_name: str,
        feature_id_i: int,
        feature_id_j: int,
    ) -> float:
        """
        Compute Pearson correlation between two SAE features based on their activation patterns
        over the token vocabulary.

        Like cosine similarity, each feature is represented as a sparse vector over token_id space
        (max activation per token). Pearson correlation additionally mean-centers each vector before
        computing the dot product, so it captures whether the two features co-vary above/below their
        respective means — rather than just whether they point in the same direction.

        For sparse SAE activations this distinction matters: cosine similarity is dominated by the
        magnitude of shared high-activation tokens, while correlation is more sensitive to whether
        the two features consistently activate together relative to their own baselines.

        Note: same token-type space limitation as ``feature_similarity_cosine_similarity`` — see TODO.md.

        Args:
            table_name: Name of the activations table.
            feature_id_i: First SAE feature dimension index.
            feature_id_j: Second SAE feature dimension index.

        Returns:
            Pearson correlation in [-1, 1]. Returns 0.0 if either feature has no activations.
        """
        CORRELATION_QUERY = f"""
        WITH agg AS (
            SELECT feature_id, token_id, MAX(activation_value) AS activation
            FROM {table_name}
            WHERE feature_id IN ({feature_id_i}, {feature_id_j})
            GROUP BY feature_id, token_id
        ),
        means AS (
            SELECT feature_id, AVG(activation) AS mean_activation
            FROM agg
            GROUP BY feature_id
        ),
        centered AS (
            SELECT a.feature_id, a.token_id, a.activation - m.mean_activation AS centered_activation
            FROM agg a
            JOIN means m ON a.feature_id = m.feature_id
        ),
        stdevs AS (
            SELECT feature_id, SQRT(SUM(centered_activation * centered_activation)) AS std
            FROM centered
            GROUP BY feature_id
        )
        SELECT
            SUM(ci.centered_activation * cj.centered_activation) / (si.std * sj.std) AS correlation
        FROM centered ci
        JOIN centered cj ON ci.token_id = cj.token_id AND ci.feature_id = {feature_id_i} AND cj.feature_id = {feature_id_j}
        JOIN stdevs si ON si.feature_id = {feature_id_i}
        JOIN stdevs sj ON sj.feature_id = {feature_id_j}
        """
        result = self.con.execute(CORRELATION_QUERY).fetchone()
        if result is None or result[0] is None:
            return 0.0
        return result[0]

    # def cluster_features(self,):
    #     """
    #     Group features by activation pattern similarity
    #     """
    # TODO: Implement this later as this is not a core method.
    #     pass

    # -------------------------------------------------------------------------
    # Interpretability
    # -------------------------------------------------------------------------

    def label_feature(self, input_df: pd.DataFrame, use_groq: bool = True) -> pd.DataFrame:
        """
        Use an LLM to propose a human-readable label for what a feature detects.

        All context strings for the feature are batched into a single LLM call to allow
        the model to identify patterns across exemplars rather than reasoning about each
        one in isolation.

        Args:
            input_df: DataFrame of top activating records for a feature. If a
                ``context_string`` column is not present, it will be generated automatically
                via ``get_context_string``.
            use_groq: If ``True`` (default), route the LLM call through Groq. If ``False``,
                call OpenAI directly.

        Returns:
            The input DataFrame with added ``llm_label`` and ``llm_reasoning``
            columns (same value repeated for every row).
        """
        if "context_string" not in input_df.columns:
            input_df = self.get_context_string(input_df)
        context_strings = input_df["context_string"].tolist()
        result = self.llm_judge_classification(context_strings, use_groq=use_groq)
        input_df["llm_label"] = result["label"]
        input_df["llm_reasoning"] = result["reasoning"]
        return input_df

    def get_token_type_breakdown(self, feature_id: int, table_name: str, sample_size: int | None = 1000) -> dict[str, dict]:
        """
        Compute the distribution of linguistic properties for tokens that activate a feature.

        Fetches activation records for the feature, reconstructs token and context text,
        then runs spaCy to classify each activating token. Aggregates counts over each property.

        Args:
            feature_id: SAE feature dimension index to analyse.
            table_name: Name of the activations table to query.
            sample_size: Maximum number of activation records to process, sampled by
                highest activation value. Defaults to 1000. Pass ``None`` to process
                all records (may OOM for high-frequency features).

        Returns:
            Dictionary mapping each property name to a ``{value: count}`` sub-dictionary.
            Property names: ``pos``, ``ner``, ``is_stop``, ``is_punct``, ``dep``,
            ``subword_position``, ``is_numeric``, ``is_upper``, ``is_title``, ``is_whitespace``.
        """
        limit_clause = f"ORDER BY activation_value DESC LIMIT {sample_size}" if sample_size is not None else ""
        FILTER_QUERY = f"""
        SELECT *
        FROM {table_name}
        WHERE feature_id = {feature_id}
        {limit_clause}
        """
        feature_df = self.con.execute(FILTER_QUERY).df()
        feature_df = self.reconstruct_token_text(feature_df)
        feature_df = self.reconstruct_context_text(feature_df)
        feature_df = self.join_token_with_context(feature_df)
        feature_df = self.classify_activating_token(feature_df)

        counters: dict[str, dict] = {
            "pos": {}, "ner": {}, "is_stop": {}, "is_punct": {},
            "dep": {}, "subword_position": {}, "is_numeric": {},
            "is_upper": {}, "is_title": {}, "is_whitespace": {},
        }
        for col in counters:
            counters[col] = feature_df[col].value_counts().to_dict()

        return counters

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def llm_judge_classification(self, context_strings: list[str], use_groq: bool = True) -> dict[str, str]:
        """
        Send a batch of context strings to an LLM and return its classification label.

        Args:
            context_strings: List of formatted context strings (e.g. from ``get_context_string``),
                each with the activating token highlighted in ``**bold**``.
            use_groq: If ``True`` (default), call through Groq using ``self.groq_model``.
                If ``False``, call OpenAI directly using ``self.openai_model``.

        Returns:
            A dict with ``label`` (parsed short label) and ``reasoning`` (the
            chain-of-thought reasoning the LLM produced before the label).
        """
        if use_groq:
            client = OpenAI(base_url=self.groq_base_url, api_key=os.getenv("GROQ_API_KEY"))
            model = self.groq_model
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = self.openai_model

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(context_strings))
        user_message = LLM_JUDGE_CLASSIFICATION_USER.format(context_strings=numbered)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_CLASSIFICATION_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content

        label = raw.strip()
        reasoning = ""
        for line in reversed(raw.strip().splitlines()):
            if line.strip().lower().startswith("label:"):
                label = line.split(":", 1)[1].strip()
                reasoning = raw[:raw.rfind(line)].strip()
                break

        return {"label": label, "reasoning": reasoning}

    def join_token_with_context(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Splice the activating token back into the middle of its context token list.

        The context window stores the surrounding tokens but replaces the center position
        with a placeholder. This method reconstructs the full surface string by inserting
        the activating ``token_text`` at position ``context_window`` in ``context_text``.

        Args:
            input_df: DataFrame with ``token_text`` (str) and ``context_text`` (list[str]) columns.

        Returns:
            The same DataFrame with an added ``token_joined_context`` column containing
            the reconstructed surface string for each row.
        """
        def build_joined(row):
            tokens = row["context_text"]
            mid = self.context_window
            return ''.join(tokens[:mid] + [row["token_text"]] + tokens[mid + 1:])

        input_df["token_joined_context"] = input_df.apply(build_joined, axis=1)
        return input_df

    def classify_activating_token(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run spaCy on reconstructed context strings to extract linguistic properties of the
        activating token.

        Uses the character offset of the activating token within ``token_joined_context``
        to locate the corresponding spaCy token and read its annotations.

        Args:
            input_df: DataFrame with ``token_text`` (str), ``context_text`` (list[str]),
                and ``token_joined_context`` (str) columns.

        Returns:
            The input DataFrame joined with new columns:
            ``pos``, ``ner``, ``is_stop``, ``is_punct``, ``dep``,
            ``subword_position``, ``is_numeric``, ``is_upper``, ``is_title``, ``is_whitespace``.
            Rows where the activating token cannot be located in the spaCy parse fall back to
            sensible defaults (``pos="X"``, ``ner="O"``, booleans derived from the raw token string).
        """
        nlp = spacy.load("en_core_web_sm")

        docs = list(nlp.pipe(input_df["token_joined_context"].tolist()))

        results = []
        for i, doc in enumerate(docs):
            row = input_df.iloc[i]
            context_tokens = row["context_text"]
            token_text = row["token_text"]

            # char offset of the activating token within token_joined_context
            char_offset = len(''.join(context_tokens[:self.context_window]))

            # find the spaCy token that covers this offset
            spacy_token = None
            for t in doc:
                if t.idx <= char_offset < t.idx + len(t.text):
                    spacy_token = t
                    break

            if spacy_token is not None:
                results.append({
                    "pos":              spacy_token.pos_,
                    "ner":              spacy_token.ent_type_ or "O",
                    "is_stop":          spacy_token.is_stop,
                    "is_punct":         spacy_token.is_punct,
                    "dep":              spacy_token.dep_,
                    "subword_position": "word_initial" if token_text.startswith(" ") else "word_medial",
                    "is_numeric":       token_text.strip().isdigit(),
                    "is_upper":         token_text.strip().isupper() and len(token_text.strip()) > 1,
                    "is_title":         token_text.strip().istitle(),
                    "is_whitespace":    token_text.strip() == "",
                })
            else:
                results.append({
                    "pos": "X", "ner": "O", "is_stop": False, "is_punct": False,
                    "dep": "", "subword_position": "word_initial" if token_text.startswith(" ") else "word_medial",
                    "is_numeric": token_text.strip().isdigit(),
                    "is_upper": token_text.strip().isupper() and len(token_text.strip()) > 1,
                    "is_title": token_text.strip().istitle(),
                    "is_whitespace": token_text.strip() == "",
                })

        return input_df.join(pd.DataFrame(results))

    def build_vocab_table(self) -> None:
        """
        Decode every token in the GPT-2 vocabulary and persist it as a DuckDB table.

        Creates (if not already present) a ``vocab`` table with columns
        ``[token_id, token_text]`` used by all token reconstruction queries.
        """
        vocab = {v: self.tokenizer.decode([v]) for v in range(self.tokenizer.vocab_size)}
        vocab_df = pd.DataFrame(vocab.items(), columns=["token_id", "token_text"])
        self.con.register("vocab_df", vocab_df)
        self.con.execute("CREATE TABLE IF NOT EXISTS vocab AS SELECT * FROM vocab_df")

    def create_features_table(self, table_name: str) -> None:
        """
        Load all Parquet shards for the HuggingFace dataset into a local DuckDB table.

        If the table already exists this method is a no-op (emits a ``UserWarning``).

        Args:
            table_name: Name to give the new DuckDB table.
        """
        if self._table_exists(table_name):
            warnings.warn(f"Table '{table_name}' already exists. Skipping creation.", UserWarning)
            return

        CREATE_TABLE_QUERY = f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM 'hf://datasets/{self.hf_dataset_path}/data/*.parquet'
         """
        self.con.execute(CREATE_TABLE_QUERY)

    def _table_exists(self, table_name: str) -> bool:
        """
        Check whether a table already exists in the ``main`` DuckDB schema.

        Args:
            table_name: Name of the table to look up.

        Returns:
            ``True`` if the table exists, ``False`` otherwise.
        """
        EXIST_QUERY = f"""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = '{table_name}'
        """
        result = self.con.execute(EXIST_QUERY).fetchone()
        return result[0] > 0

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from the DuckDB database if it exists.

        Args:
            table_name: Name of the table to drop.
        """
        DROP_QUERY = f"""
        DROP TABLE IF EXISTS {table_name}
        """
        self.con.execute(DROP_QUERY)

    def drop_column(self, table_name: str, column_name: str) -> None:
        """
        Remove a column from an existing DuckDB table.

        Args:
            table_name: Name of the table to alter.
            column_name: Name of the column to drop.
        """
        DROP_COL_QUERY = f"""
        ALTER TABLE {table_name} DROP COLUMN {column_name}
        """
        self.con.execute(DROP_COL_QUERY)

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute an arbitrary SQL statement against the DuckDB connection.

        Args:
            sql: SQL string to execute.

        Returns:
            Query results as a pandas DataFrame.
        """
        return self.con.execute(sql).df()

    def close(self) -> None:
        """Close the DuckDB connection, flushing any pending writes."""
        self.con.close()


