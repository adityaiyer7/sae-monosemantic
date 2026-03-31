"""
HuggingFace-backed DuckDB connector for the SAE feature explorer.

Creates in-memory DuckDB connections with lazy VIEWs over HF parquets,
so no local data download is required.
"""
import logging
import sys
from pathlib import Path
import duckdb
import pandas as pd

# Ensure project root is on sys.path so src/ imports work
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.feature_visualization import FeatureAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HF_USER = "thedarkknight7"

# All 16 SAE configurations.
# key: human-readable label shown in the UI
# value: dict with expansion factor, lambda value, sampling flag, and derived fields
CONFIGS: dict[str, dict] = {}

for _exp in [4, 8, 16, 32]:
    for _lam, _lam_str in [(1e-2, "0.01"), (1e-4, "0.0001")]:
        for _sampling in [False, True]:
            _sampling_label = "sampling" if _sampling else "no sampling"
            _key = f"{_exp}x | λ={_lam_str} | {_sampling_label}"
            _hf_suffix = f"{_exp}x_{_lam_str}" + ("_sampling" if _sampling else "")
            CONFIGS[_key] = {
                "expansion": _exp,
                "lam": _lam,
                "lam_str": _lam_str,
                "sampling": _sampling,
                "hf_path": f"{HF_USER}/SAE_monosemanticity_features_{_hf_suffix}",
                "num_features": _exp * 768,
            }

# Sorted for consistent ordering in the UI
CONFIGS = dict(sorted(CONFIGS.items(), key=lambda x: (
    x[1]["expansion"], x[1]["lam"], x[1]["sampling"]
)))

# The DuckDB view/table name used within each analyzer instance
FEATURES_TABLE = "features"


class HFFeatureAnalyzer(FeatureAnalyzer):
    """
    Subclass of FeatureAnalyzer backed by an in-memory DuckDB with a lazy VIEW
    over HuggingFace parquets. No local data download required — DuckDB pushes
    predicates down to only read relevant parquet row groups on demand.
    """

    def __init__(
        self,
        hf_dataset_path: str,
        expansion_factor: int,
        hf_token: str,
        context_window: int = 10,
        model_hidden_dim_size: int = 768,
    ) -> None:
        # Bypass FeatureAnalyzer.__init__ — we use in-memory DuckDB instead of a file
        from transformers import GPT2Tokenizer

        self.hf_dataset_path = hf_dataset_path
        self.expansion_factor = expansion_factor
        self.model_hidden_dim_size = model_hidden_dim_size
        self.context_window = context_window

        log.info("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        log.info("Tokenizer loaded.")

        log.info("Connecting to DuckDB (in-memory)...")
        self.con = duckdb.connect(":memory:")
        self.con.execute("DROP SECRET IF EXISTS hf_token")
        self.con.execute(
            f"CREATE SECRET hf_token (TYPE huggingface, TOKEN '{hf_token}')"
        )
        log.info("DuckDB connected. Registering HuggingFace secret.")

        # LLM routing attributes (mirrors FeatureAnalyzer defaults)
        self.groq_model = "openai/gpt-oss-120b"
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.openai_model = "gpt-5.4"

        log.info("Building vocab table (%d tokens)...", self.tokenizer.vocab_size)
        self.build_vocab_table()
        log.info("Vocab table ready.")

        log.info("Creating lazy VIEW over HF parquets: %s", hf_dataset_path)
        self.con.execute(
            f"CREATE VIEW {FEATURES_TABLE} AS "
            f"SELECT * FROM 'hf://datasets/{hf_dataset_path}/data/*.parquet'"
        )
        log.info("VIEW created. Analyzer ready — queries will fetch parquet row groups on demand.")

    def get_dead_features(self) -> pd.DataFrame:
        """Return feature IDs that never fired (not present in the parquets)."""
        alive_df = self.con.execute(
            f"SELECT DISTINCT feature_id FROM {FEATURES_TABLE}"
        ).df()
        alive = set(alive_df["feature_id"].tolist())
        total = self.expansion_factor * self.model_hidden_dim_size
        dead = sorted(set(range(total)) - alive)
        return pd.DataFrame({"feature_id": dead})

    def get_activation_values(self, table_name: str, feature_id: int) -> pd.DataFrame:
        """Return all activation_value rows for a given feature (used for histogram)."""
        return self.con.execute(f"""
            SELECT activation_value FROM {table_name}
            WHERE feature_id = {feature_id}
        """).df()

    def get_activation_stats(self, table_name: str, feature_id: int) -> dict:
        """Return summary statistics for a feature's activation distribution."""
        df = self.get_activation_values(table_name, feature_id)
        if df.empty:
            return {}
        vals = df["activation_value"]
        return {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
            "count": int(len(vals)),
            "unique_tokens": int(
                self.con.execute(f"""
                    SELECT COUNT(DISTINCT token_id) FROM {table_name}
                    WHERE feature_id = {feature_id}
                """).fetchone()[0]
            ),
        }
