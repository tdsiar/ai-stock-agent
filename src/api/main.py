import os
import sys
import yaml
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TICKERS       = config["tickers"]
PROCESSED_DIR = os.path.join(BASE_DIR, config["data"]["processed_dir"])
DB_PATH       = os.path.join(BASE_DIR, config["data"]["db_path"])
STATIC_DIR    = os.path.join(BASE_DIR, "src", "api", "static")

# Make backtest module importable
sys.path.insert(0, BASE_DIR)
from src.backtest.engine import backtest_ticker

app = FastAPI(title="AI Stock Agent API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Routes
@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS}


@app.get("/backtest/{ticker}")
def get_backtest(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker} not in configured tickers")
    conn   = sqlite3.connect(DB_PATH)
    result = backtest_ticker(ticker, conn)
    conn.close()
    return result


@app.get("/data/{ticker}")
def get_ticker_data(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker} not in configured tickers")

    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No processed data found for {ticker}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index()
    df.columns = ["date" if i == 0 else c for i, c in enumerate(df.columns)]
    return df.to_dict(orient="records")
