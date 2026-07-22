from pathlib import Path


# =============================================================================
# Project directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOCAL_OUTPUT_DIR = OUTPUT_DIR / "local"
SUBMISSION_OUTPUT_DIR = OUTPUT_DIR / "submissions"

REPORT_DIR = PROJECT_ROOT / "reports"
ASSET_DIR = PROJECT_ROOT / "assets"


# =============================================================================
# Raw dataset files
# =============================================================================

INTERACTIONS_FILE = RAW_DATA_DIR / "lfm-challenge.inter_train"
ITEMS_FILE = RAW_DATA_DIR / "lfm-challenge.item"
USERS_FILE = RAW_DATA_DIR / "lfm-challenge.user"
MUSICNN_FILE = RAW_DATA_DIR / "lfm-challenge.musicnn"

LOCAL_TRAIN_FILE = RAW_DATA_DIR / "local_train.tsv"
LOCAL_VAL_FILE = RAW_DATA_DIR / "local_val.tsv"


# =============================================================================
# Experiment settings
# =============================================================================

RANDOM_SEED = 42
TOP_K = 10

USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"

INTERACTION_COUNT_COLUMN = "count"
LOCAL_EVENT_COLUMN = "events"