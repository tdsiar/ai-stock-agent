import os
import yaml
import sqlite3
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# Features the model looks at
FEATURES = ["ma_7", "ma_30", "momentum_10", "volume_ma_20"]


# Train and evaluate one ticker
def run_model(ticker: str, conn: sqlite3.Connection) -> dict:
    df = pd.read_sql(f"SELECT * FROM {ticker}", conn, index_col="date", parse_dates=["date"])

    # Target: will price be higher tomorrow than today? (1 = up, 0 = down)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = df["target"]

    # 80% of data for training, 20% for testing — in time order, no shuffling
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict — values close to 1 = predicts up, close to 0 = predicts down
    y_pred = model.predict(X_test)
    y_pred_direction = (y_pred >= 0.5).astype(int)

    # Evaluate
    mae               = mean_absolute_error(y_test, y_pred)
    directional_acc   = (y_pred_direction == y_test).mean() * 100

    log.info(f"{ticker}: MAE={mae:.4f} | Directional Accuracy={directional_acc:.1f}%")

    return {
        "ticker":              ticker,
        "mae":                 round(mae, 4),
        "directional_accuracy": round(directional_acc, 1),
        "test_days":           len(y_test),
    }


# Main
if __name__ == "__main__":
    log.info("=== Linear Regression model started ===")

    conn    = sqlite3.connect(DB_PATH)
    results = []

    for ticker in TICKERS:
        result = run_model(ticker, conn)
        results.append(result)

    conn.close()

    # Summary table
    print("\n── Results Summary ──────────────────────────────")
    print(f"{'Ticker':<8} {'MAE':>8} {'Dir. Accuracy':>15} {'Test Days':>10}")
    print("-" * 45)
    for r in results:
        print(f"{r['ticker']:<8} {r['mae']:>8} {r['directional_accuracy']:>14}% {r['test_days']:>10}")

    avg_acc = sum(r["directional_accuracy"] for r in results) / len(results)
    print(f"\nAverage directional accuracy: {avg_acc:.1f}%")
    log.info("=== Linear Regression model complete ===")
