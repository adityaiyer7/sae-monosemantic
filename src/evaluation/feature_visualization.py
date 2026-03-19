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


# Note: feature_id is the SAE feature dimension index


user = whoami(token=os.getenv("HF_TOKEN"))


class FeatureAnalyzer:
    def __init__(self, HF_dataset_path: str, db_name: str, expansion_factor: int, model_hidden_dim_size:int = 768):
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
        self.build_vocab_table()
    
    """
    Token and context Reconstruction Methods
    """
    def reconstruct_token_text(self, df: pd.DataFrame) -> pd.DataFrame:
        self.con.register("_input", df)
        result = self.con.execute("""
            SELECT s.*, v.token_text
            FROM _input s
            JOIN vocab v ON s.token_id = v.token_id
        """).df()
        self.con.unregister("_input")
        return result
    
    def reconstruct_context_text(self, df:pd.DataFrame) -> pd.DataFrame:
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
        Join the context text array into a single readable string with the activating token highlighted
        This method assumes, reconstruct_context_text and reconstruct_token_text have already been called before
        """
        ACTIVATING_TOKEN_IDX  = 10

        def build_string(row):
            context_list = []
            for idx, text in enumerate(row["context_text"]):
                if idx == ACTIVATING_TOKEN_IDX:
                    context_list.append(f"**{row['token_text']}**")
                else:
                    context_list.append(text)
            return ''.join(context_list)

        df["context_string"] = df.apply(build_string, axis=1)
        return df
                        

    
    """
    Per Feature Analysis
    """
    
    
    def get_top_activations(self, table_name: str, feature_id: int, top_k: int, sort_order:str = 'descending') -> pd.DataFrame:
        """
        For a given feature ID, return the top-k activating tokens ranked by activation value, by default its sorted in descending order
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
    

    def get_most_active_features(self, table_name:str):
        """
        This method gives us the number of tokens that activate per dimension
        """
        NUM_FEATURES_QUERY = f"""
            SELECT feature_id, COUNT(*) AS num_features
            FROM {table_name}
            GROUP BY feature_id
            ORDER BY num_features DESC
        """
        return self.con.execute(NUM_FEATURES_QUERY).df()
    
    def get_feature_density(self, table_name:str, num_unique_tokens:int = 50257):
        """
        This gives us the fraction of tokens that activate this feature relative to all activation events
        In simple terms, "what fraction of all activation events activate this feature"
        Idea here is that out of all tokens that caused some activation (above our threshold set in intepretability.py), what fraction activated this feature?
        """
        FEATURE_DENSITY_QUERY = f"""
            SELECT feature_id, COUNT(*)/(SELECT COUNT(*) FROM {table_name}) AS feature_density
            FROM {table_name}
            GROUP BY feature_id
        """
        return self.con.execute(FEATURE_DENSITY_QUERY).df()


    def get_activation_distribution(self, table_name:str, feature_id: int, save_figs:bool = False, figs_dir: str = "figs"):
        """
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
        activation_value_25th_percentile = filtered_df["activation_value"].quantile(q = 0.25)
        activation_value_75th_percentile = filtered_df["activation_value"].quantile(q = 0.75)

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
            "mean_activation_score" : mean_activation_score,
            "median_activation_score": median_activation_score,
            "standard_deviatin_activation_scores": standard_deviatin_activation_scores,
            "activation_value_25th_percentile": activation_value_25th_percentile,
            "activation_value_75th_percentile":activation_value_75th_percentile,
            "unique_token_id_count": unique_token_id_count
        }
    
    def get_activation_distribution_per_token_id(self, table_name:str, token_id: int):
        """
        """
        # for each unique token_id (that could appear in different contexts), calculate the following
        # get number of features that activate for the token
        # get all feature_ids that activate
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
            "num_features_activated_by_token" : num_features_activated_by_token
        }
    
    def get_co_occuring_features(self, table_name: str, feature_id: int, activation_percentile: float = 0.75):
        """
        For a given feature, find which other features frequently fire on the same tokens.

        Only tokens whose activation value for feature_id meets or exceeds the specified
        percentile threshold are considered — this avoids polluting results with tokens
        that only marginally activate the feature. 
        
        Also, we chose to take the percentile over a fixed number to make it scale invariant. 

        Args:
            table_name: Name of the activations table.
            feature_id: The SAE feature dimension index to analyse.
            activation_percentile: Percentile (0–1) used to threshold activations for
                feature_id before computing co-occurrences. Defaults to 0.75 (top quartile).

        Returns:
            DataFrame with columns [feature_id, co_occurrence_count], sorted descending
            by co_occurrence_count (number of distinct tokens shared with feature_id).
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


    """
    Cross Feature Analysis
    """
    def get_dead_features(self, table_name:str):
        # TODO - this method has to be re-written following the changes in @interpretability.py
        """
        This gives us the feature_id's (dimensions) that didn't fire for any input. 
        """
        total_num_features = self.expansion_factor * self.model_hidden_dim_size
        GET_DEAD_FEATURES = f"""
        SELECT gs.feature_id
        FROM generate_series(0, {total_num_features - 1}) AS gs(feature_id)
        LEFT JOIN {table_name} t
        ON t.feature_id = gs.feature_id
        WHERE t.feature_id IS NULL
        """
        return self.con.execute(GET_DEAD_FEATURES).df()
    
    def feature_similarity_cosine_similarity(self, table_name: str, feature_id_i: int, feature_id_j: int) -> float:
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
            SUM(a.activation * b.activation) / (na.norm * nb.norm) AS cosine_similarity
        FROM agg a
        JOIN agg b ON a.token_id = b.token_id AND a.feature_id = {feature_id_i} AND b.feature_id = {feature_id_j}
        JOIN norms na ON na.feature_id = {feature_id_i}
        JOIN norms nb ON nb.feature_id = {feature_id_j}
        """
        result = self.con.execute(COS_SIM_QUERY).fetchone()
        if result is None or result[0] is None:
            return 0.0
        return result[0]

    def feature_similarity_correlation(self, table_name: str, feature_id_i: int, feature_id_j: int) -> float:
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

        Note: same token-type space limitation as feature_similarity_cosine_similarity — see TODO.md.

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

    
    """
    Interpretability
    """

    def get_feature_exemplars(self,):
        """
        Return top-k full context windows for a feature, showing surrounding tokens (this is what makes features interpretable)
        """
        #TODO
        pass
    
    def label_feature(self,):
        """
        Given exemplars, use an LLM propose a human-readable label for what the feature detects
        """
        #TODO
        pass
    
    def get_token_type_breakdown(self,feature_id:int, table_name:str):
        """
        For a feature, show distribution over POS tags, punctuation, named entities, etc.
        """
        #TODO
        FILTER_QUERY = f"""
        SELECT * 
        FROM {table_name}
        WHERE feature_id = {feature_id}
        """
        feature_df = self.con.execute(FILTER_QUERY).df()
        feature_df = self.con.reconstruct_token_text(feature_df)
        feature_df = self.con.reconstruct_context_text(feature_df)
        feature_df["joined_context"] = ''.join(feature_df('context_text'))
        


        # Query all rows for feature_id from the table
        # Call reconstruct_token_text → get token_text
        # Call reconstruct_context_text → get context_text (list of 21 tokens)
        # ''.join(context_text) → full sentence (GPT-2 tokens already have spaces baked in, so this produces valid text)
        # Run spaCy on that string
        # Find the activating token by character offset: len(''.join(context_text[:10])) gives you the exact start position (since ACTIVATING_TOKEN_IDX = 10 is hardcoded at feature_visualization.py:69)
        # Use doc.char_span() or iterate over doc to find which spaCy token covers that offset


        # Return type:
        #{
        # "pos": {"NOUN": 45, "VERB": 12, ...},
        # "ner": {"PERSON": 8, "O": 80, ...},
        # "is_stop": {True: 30, False: 70},
        # "subword_position": {"word_initial": 60, "word_medial": 40},
        # ...
        # }

        # Classes 
            # spaCy (free, from context):

            # POS tag (coarse): NOUN, VERB, ADJ, ADV, DET, PREP, PRON, PUNCT, NUM, X
            # NER label: PERSON, ORG, GPE, DATE, MONEY, O (not an entity)
            # Is stop word
            # Is punctuation
            # Dependency role: nsubj, dobj, ROOT, amod, det, etc.
            # Derived without spaCy (from token string alone):

            # Subword position: word-initial (leading space) vs. word-medial
            # Is numeric (all digits)
            # Is all-caps
            # Is title-case
            # Is whitespace/special character



    """
    Utility Methods
    """
    def build_vocab_table(self):
        vocab = {v: self.tokenizer.decode([v]) for v in range(self.tokenizer.vocab_size)}
        vocab_df = pd.DataFrame(vocab.items(), columns=["token_id", "token_text"])
        self.con.register("vocab_df", vocab_df)
        self.con.execute("CREATE TABLE IF NOT EXISTS vocab AS SELECT * FROM vocab_df")

    def create_features_table(self, table_name: str) -> None:
        if self._table_exists(table_name):
            warnings.warn(f"Table '{table_name}' already exists. Skipping creation.", UserWarning)
            return
        
        CREATE_TABLE_QUERY = f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM 'hf://datasets/{self.hf_dataset_path}/data/*.parquet'
         """
        self.con.execute(CREATE_TABLE_QUERY)

    def _table_exists(self, table_name: str) -> bool:
        EXIST_QUERY = f"""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = '{table_name}'
        """
        result = self.con.execute(EXIST_QUERY).fetchone()
        return result[0] > 0
    
    def drop_table(self, table_name: str):
        DROP_QUERY = f"""
        DROP TABLE IF EXISTS {table_name}
        """
        self.con.execute(DROP_QUERY)
    
    def drop_column(self, table_name: str, column_name:str):
        DROP_COL_QUERY = f"""
        ALTER TABLE {table_name} DROP COLUMN {column_name}
        """
        self.con.execute(DROP_COL_QUERY)

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def close(self) -> None:
        self.con.close()




def main(expansion_factor: int, _lambda: float):
    project_root = Path(__file__).parents[2]
    load_dotenv(project_root / ".env")

    HF_dataset_path = f"thedarkknight7/SAE_monosemanticity_features_{expansion_factor}x_{_lambda}"
    table_name=f"hf_{expansion_factor}x_{str(_lambda).replace('.', '_')}_full"

    feature_analyzer = FeatureAnalyzer(
        HF_dataset_path = HF_dataset_path,
        db_name = "hf_trial",
        expansion_factor = expansion_factor
    )
    feature_analyzer.create_features_table(table_name = table_name)

    


if __name__ == "__main__":    
    main(expansion_factor=8, _lambda=1e-4)
