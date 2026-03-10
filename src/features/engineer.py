import os
import yaml
import logging
import pandas as pd

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TICKERS       = config["tickers"]
RAW_DIR       = os.path.join(BASE_DIR, config["data"]["raw_dir"])
PROCESSED_DIR = os.path.join(BASE_DIR, config["data"]["processed_dir"])
LOG_DIR       = os.path.join(BASE_DIR, config["logging"]["log_dir"])
LOG_FILE      = os.path.join(LOG_DIR, config["logging"]["log_file"])

# Logging
os.makedirs(PROCESSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Feature engineering
def engineer_features(ticker: str) -> None:
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    out_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")

    if not os.path.exists(raw_path):
        log.warning(f"{ticker}: raw CSV not found, skipping")
        return

    df = pd.read_csv(raw_path, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]

    # Moving averages
    df["ma_7"]  = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()

    # Momentum — percentage change over 10 days
    df["momentum_10"] = df["Close"].pct_change(periods=10) * 100

    # Volume trend — 20-day moving average of volume
    df["volume_ma_20"] = df["Volume"].rolling(window=20).mean()

    df.dropna(inplace=True)
    df.to_csv(out_path)
    log.info(f"{ticker}: features saved → {out_path} ({len(df)} rows)")


# Main
if __name__ == "__main__":
    log.info("=== Feature engineering started ===")
    for ticker in TICKERS:
        engineer_features(ticker)
    log.info("=== Feature engineering complete ===")
