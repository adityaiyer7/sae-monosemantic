import os
import warnings
import duckdb
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files
from huggingface_hub import login
from huggingface_hub import whoami
from transformers import GPT2Tokenizer


# Note: feature_id is the SAE feature dimension index


user = whoami(token=os.getenv("HF_TOKEN"))


class FeatureAnalyzer:
    def __init__(self, HF_dataset_path: str, db_name: str, expansion_factor: int, model_hidden_dim_size:int = 768):
        self.hf_dataset_path = HF_dataset_path
        self.db_name = db_name
        self.con = duckdb.connect(f'{self.db_name}.db')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.con.execute(f"""
            CREATE SECRET IF NOT EXISTS hf_token (TYPE huggingface, TOKEN '{os.getenv("HF_TOKEN")}')
        """)
        self.expansion_factor = expansion_factor
        self.model_hidden_dim_size = model_hidden_dim_size
        # self.build_vocab_table()
    
    # core analysis methods
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
        Idea here is that out of all tokens that caused some activation (above our threshold), what fraction activated this feature?
        """
        FEATURE_DENSITY_QUERY = f"""
            SELECT feature_id, COUNT(*)/(SELECT COUNT(*) FROM {table_name}) AS feature_density
            FROM {table_name}
            GROUP BY feature_id
        """
        return self.con.execute(FEATURE_DENSITY_QUERY).df()


    def get_activation_distribution(self, table_name:str, feature_id: int, save_figs:bool = False):
        """
        """

        # get number of unique tokens represented by in this dimension
        # get mean activations
        # get standard deviation of activations
        # get percentiles of activation values
        # TODO
        pass
    
    def get_activation_distribution_per_token_id(self, table_name:str, feature_id: int, save_figs:bool = False):
        """
        """
        # for each unique token_id (that could appear in different contexts), calculate the following
        # get mean activations
        # get standard deviation of activations
        # get percentiles of activation values
        # TODO
        pass
    
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

    # Utility Methods
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
    



def main():
    feature_analyzer = FeatureAnalyzer(
        HF_dataset_path = "thedarkknight7/SAE_monosemanticity_features_4x",
        db_name = "hf_trial",
        expansion_factor = 4
    )
    feature_analyzer.create_features_table(table_name="hf_4x_full")
    print("created full database from HF!")
    print(feature_analyzer.get_dead_features(table_name = "hf_4x_full"))
    # top_activations_df = feature_analyzer.get_top_activations(table_name = "hf_trial_table", feature_id = 3053, top_k = 25)
    # reconstructed_df = feature_analyzer.reconstruct_context_text(df=top_activations_df)
    # reconstructed_df = feature_analyzer.reconstruct_token_text(df = reconstructed_df)
    # print(reconstructed_df)




    # feature_analyzer.con.execute("ALTER TABLE hf_trial_table DROP COLUMN token_text")
    # print("dropped existing token text column")
    #feature_analyzer.con.execute("ALTER TABLE hf_trial_table DROP COLUMN context_text")
    # print("dropped existing context text column")


    # print("printing token")
    # print(feature_analyzer.reconsturct_token_text(table_name = "hf_trial_table"))


    #print(feature_analyzer.reconstruct_context_text(table_name = "hf_16x_full"))


    # python -c "import duckdb; con = duckdb.connect('hf_trial.db'); con.execute('DROP TABLE IF EXISTS hf_16x_full'); print('done')"
    


if __name__ == "__main__":    
    main()
