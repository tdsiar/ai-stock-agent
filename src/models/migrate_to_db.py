import os
import yaml
import sqlite3
import logging
import pandas as pd

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TICKERS       = config["tickers"]
PROCESSED_DIR = os.path.join(BASE_DIR, config["data"]["processed_dir"])
DB_PATH       = os.path.join(BASE_DIR, config["data"]["db_path"])
LOG_DIR       = os.path.join(BASE_DIR, config["logging"]["log_dir"])
LOG_FILE      = os.path.join(LOG_DIR, config["logging"]["log_file"])

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Migrate one ticker
def migrate_ticker(ticker: str, conn: sqlite3.Connection) -> None:
    csv_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")

    if not os.path.exists(csv_path):
        log.warning(f"{ticker}: processed CSV not found, skipping")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df["ticker"] = ticker

    # Write to SQLite — if table exists, replace it cleanly
    df.to_sql(ticker, conn, if_exists="replace", index=True)
    log.info(f"{ticker}: migrated {len(df)} rows to SQLite")


# Main
if __name__ == "__main__":
    log.info("=== SQLite migration started ===")

    conn = sqlite3.connect(DB_PATH)

    for ticker in TICKERS:
        migrate_ticker(ticker, conn)

    conn.close()
    log.info(f"=== Migration complete — database at {DB_PATH} ===")
