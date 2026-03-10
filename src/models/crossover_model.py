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

TICKERS  = config["tickers"]
DB_PATH  = os.path.join(BASE_DIR, config["data"]["db_path"])
LOG_DIR  = os.path.join(BASE_DIR, config["logging"]["log_dir"])
LOG_FILE = os.path.join(LOG_DIR, config["logging"]["log_file"])

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


# Evaluate one ticker
def run_crossover(ticker: str, conn: sqlite3.Connection) -> dict:
    df = pd.read_sql(f"SELECT * FROM {ticker}", conn, index_col="date", parse_dates=["date"])

    # The signal: 1 when MA7 is above MA30 (predict up), 0 when below (predict down)
    df["signal"] = (df["ma_7"] > df["ma_30"]).astype(int)

    # What actually happened the next day
    df["actual"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    # Only evaluate on the same 20% test window as linear regression
    split = int(len(df) * 0.8)
    test = df.iloc[split:]

    directional_acc = (test["signal"] == test["actual"]).mean() * 100

    log.info(f"{ticker}: Directional Accuracy={directional_acc:.1f}%")

    return {
        "ticker": ticker,
        "directional_accuracy": round(directional_acc, 1),
        "test_days": len(test),
    }


# Main
if __name__ == "__main__":
    log.info("=== MA Crossover model started ===")

    conn    = sqlite3.connect(DB_PATH)
    results = []

    for ticker in TICKERS:
        result = run_crossover(ticker, conn)
        results.append(result)

    conn.close()

    print("\n── Results Summary ──────────────────────────────")
    print(f"{'Ticker':<8} {'Dir. Accuracy':>15} {'Test Days':>10}")
    print("-" * 37)
    for r in results:
        print(f"{r['ticker']:<8} {r['directional_accuracy']:>14}% {r['test_days']:>10}")

    avg_acc = sum(r["directional_accuracy"] for r in results) / len(results)
    print(f"\nAverage directional accuracy: {avg_acc:.1f}%")
    log.info("=== MA Crossover model complete ===")
