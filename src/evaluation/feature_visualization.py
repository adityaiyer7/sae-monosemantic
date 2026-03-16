import os
import warnings
import duckdb
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files
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


    """
    Cross Feature Analysis
    """
    def get_dead_features(self, table_name:str):
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
    print("created full database from HF!")
    # top_activations_df = feature_analyzer.get_top_activations(table_name = table_name, feature_id = 3000, top_k = 25)
    # reconstructed_df = feature_analyzer.reconstruct_context_text(df=top_activations_df)
    # reconstructed_df = feature_analyzer.reconstruct_token_text(df = reconstructed_df)
    # reconstructed_df = feature_analyzer.get_context_string(df = reconstructed_df)
    # # activation_distribution_res = feature_analyzer.get_activation_distribution(table_name = table_name, feature_id = 3000)
    # token_id = feature_analyzer.tokenizer.encode("love")[0]
    # res = feature_analyzer.get_activation_distribution_per_token_id(table_name = table_name, token_id = token_id)

    # print(res)

    dead_features = feature_analyzer.get_dead_features(table_name = table_name)
    print(dead_features)


    # feature_analyzer.con.execute(f"ALTER TABLE {table_name} DROP COLUMN token_text")
    # print("dropped existing token text column")
    #feature_analyzer.con.execute(f"ALTER TABLE {table_name} DROP COLUMN context_text")
    # print("dropped existing context text column")



    # python -c "import duckdb; con = duckdb.connect('hf_trial.db'); con.execute('DROP TABLE IF EXISTS hf_16x_full'); print('done')"
    


if __name__ == "__main__":    
    main(expansion_factor=8, _lambda=1e-4)
