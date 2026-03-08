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

all_files = [
    f for f in list_repo_files("thedarkknight7/SAE_monosemanticity_features_4x", repo_type = "dataset")
    if f.endswith(".parquet")
]
sample_files = all_files[:5]
print(sample_files)

dataset = load_dataset("thedarkknight7/SAE_monosemanticity_features_4x", data_files = sample_files, verification_mode = "no_checks")

print(dataset)

# db_name = "hf_trial"
# con = duckdb.connect(f"{db_name}.db")
# table_name = "hf_trial_table"
# con.register("temp_view", dataset["train"].data.table)
# CREATE_TABLE_QUERY = f"""
#             CREATE TABLE {table_name} AS
#             SELECT * FROM  temp_view
#          """
# con.execute(CREATE_TABLE_QUERY)


# print(con.execute(f"SELECT * FROM {table_name} LIMIT 10").df())

class FeatureAnalyzer:
    def __init__(self, HF_dataset_path: str, db_name: str):
        self.hf_dataset_path = HF_dataset_path
        self.db_name = db_name
        self.con = duckdb.connect(f'{self.db_name}.db')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.build_vocab_table()
    
    # core analysis methods
    def reconsturct_token_text(self, table_name):
        ADD_COLUMN_QUERY = f"ALTER TABLE {table_name} ADD COLUMN token_text VARCHAR"
        RECONSTRUCT_TOKEN_TEXT_QUERY = f"""
            UPDATE {table_name} t
            SET token_text = v.token_text
            FROM vocab v
            WHERE t.token_id = v.token_id
        """
        self.con.execute(ADD_COLUMN_QUERY)
        self.con.execute(RECONSTRUCT_TOKEN_TEXT_QUERY)
        return self.con.execute(f"SELECT * FROM {table_name} LIMIT 10").df()

    def reconstruct_context_text(self, table_name):
        ADD_COLUMN_QUERY = f"ALTER TABLE {table_name} ADD COLUMN context_text VARCHAR[]"
        RECONSTRUCT_CONTEXT_QUERY = f"""
            UPDATE {table_name} t
            SET context_text = (
                SELECT list(v.token_text ORDER BY pos)
                FROM unnest(t.context_token_ids) WITH ORDINALITY AS u(tid, pos)
                JOIN vocab v ON u.tid = v.token_id
            )
        """
        self.con.execute(ADD_COLUMN_QUERY)
        self.con.execute(RECONSTRUCT_CONTEXT_QUERY)
        return self.con.execute(f"SELECT * FROM {table_name} LIMIT 10").df()

    
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
            SELECT * FROM {self.hf_dataset_path} 
         """
        self.con.execute(CREATE_TABLE_QUERY)

    def _table_exists(self, table_name: str) -> bool:
        EXIST_QUERY = f"""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = {table_name}
        """
        result = self.con.execute(EXIST_QUERY).fetchone()
        return result[0] > 0
    



def main():
    feature_analyzer = FeatureAnalyzer(
        HF_dataset_path = "thedarkknight7/SAE_monosemanticity_features_4x",
        db_name = "hf_trial"
    )
    feature_analyzer.con.execute("ALTER TABLE hf_trial_table DROP COLUMN token_text")
    print("dropped existing token text column")
    feature_analyzer.con.execute("ALTER TABLE hf_trial_table DROP COLUMN context_text")
    print("dropped existing context text column")


    print("printing token")
    print(feature_analyzer.reconsturct_token_text(table_name = "hf_trial_table"))
    print(feature_analyzer.reconstruct_context_text(table_name = "hf_trial_table"))
    


if __name__ == "__main__":    
    main()



# pq_file_path = 'src/evaluation/features/features_4x/chunk_0003.parquet'