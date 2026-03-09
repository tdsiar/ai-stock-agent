import os
import yaml
import logging
import yfinance as yf
import pandas as pd
from datetime import date, datetime

# ── Load config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TICKERS   = config["tickers"]
START     = config["data"]["start_date"]
END       = date.today().isoformat() if config["data"]["end_date"] == "today" else config["data"]["end_date"]
RAW_DIR   = os.path.join(BASE_DIR, config["data"]["raw_dir"])
LOG_DIR   = os.path.join(BASE_DIR, config["logging"]["log_dir"])
LOG_FILE  = os.path.join(LOG_DIR, config["logging"]["log_file"])

# ── Set up logging ────────────────────────────────────────────────────────────
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),          # also print to terminal
    ],
)
log = logging.getLogger(__name__)


# ── Cache check ───────────────────────────────────────────────────────────────
def already_downloaded_today(csv_path: str) -> bool:
    """Return True if the file exists and was last modified today."""
    if not os.path.exists(csv_path):
        return False
    modified = datetime.fromtimestamp(os.path.getmtime(csv_path)).date()
    return modified == date.today()


# ── Fetch one ticker ──────────────────────────────────────────────────────────
def fetch_ticker(ticker: str) -> None:
    csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")

    if already_downloaded_today(csv_path):
        log.info(f"{ticker}: cache hit — skipping API call")
        return

    log.info(f"{ticker}: downloading {START} → {END}")
    try:
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

        if df.empty:
            log.warning(f"{ticker}: no data returned")
            return

        df.to_csv(csv_path)
        log.info(f"{ticker}: saved {len(df)} rows → {csv_path}")

    except Exception as e:
        log.error(f"{ticker}: failed — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Pipeline run started ===")
    for ticker in TICKERS:
        fetch_ticker(ticker)
    log.info("=== Pipeline run complete ===")
