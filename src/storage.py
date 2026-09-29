import os
from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
HISTORY_FILE = DATA_DIR / "history.csv"
COLUMNS = ["case_id", "age", "gender", "result", "confidence", "raw_score", "timestamp"]


def append_prediction(record: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {key: record.get(key, "") for key in COLUMNS}
    exists = HISTORY_FILE.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(HISTORY_FILE, mode="a", header=not exists, index=False)


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_csv(HISTORY_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
