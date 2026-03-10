import os
import yaml
import sqlite3
import pandas as pd

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TICKERS = config["tickers"]
DB_PATH = os.path.join(BASE_DIR, config["data"]["db_path"])


# Performance metrics

def total_return(portfolio_values: list) -> float:
    # How much did we make or lose overall as a percentage
    return round((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100, 2)


def max_drawdown(portfolio_values: list) -> float:
    # Worst peak-to-trough drop — measures how painful the ride was
    peak     = portfolio_values[0]
    max_dd   = 0.0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    return round(max_dd, 2)


def sharpe_ratio(portfolio_values: list) -> float:
    # Return relative to risk — higher is better, above 1.0 is decent
    series  = pd.Series(portfolio_values)
    returns = series.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    # Annualized: multiply daily sharpe by sqrt(252 trading days)
    return round((returns.mean() / returns.std()) * (252 ** 0.5), 2)


# Strategies

def strategy_buy_and_hold(df: pd.DataFrame, starting_cash: float) -> list:
    # Buy on day 1, never sell — the baseline every strategy must beat
    shares = starting_cash / df["Close"].iloc[0]
    return [shares * price for price in df["Close"]]


def strategy_ma_crossover(df: pd.DataFrame, starting_cash: float) -> list:
    # Buy when MA7 crosses above MA30, sell when it crosses below
    cash        = starting_cash
    shares      = 0.0
    portfolio   = []

    for _, row in df.iterrows():
        if row["ma_7"] > row["ma_30"] and shares == 0:
            # Signal: buy
            shares = cash / row["Close"]
            cash   = 0.0
        elif row["ma_7"] < row["ma_30"] and shares > 0:
            # Signal: sell
            cash   = shares * row["Close"]
            shares = 0.0

        portfolio.append(cash + shares * row["Close"])

    return portfolio


def strategy_rsi(df: pd.DataFrame, starting_cash: float) -> list:
    # Buy when RSI drops below 30 (oversold), sell when RSI rises above 70 (overbought)
    cash      = starting_cash
    shares    = 0.0
    portfolio = []

    for _, row in df.iterrows():
        if row["rsi_14"] < 30 and shares == 0:
            # Oversold — buy
            shares = cash / row["Close"]
            cash   = 0.0
        elif row["rsi_14"] > 70 and shares > 0:
            # Overbought — sell
            cash   = shares * row["Close"]
            shares = 0.0

        portfolio.append(cash + shares * row["Close"])

    return portfolio


# Run backtest for one ticker

def backtest_ticker(ticker: str, conn: sqlite3.Connection, starting_cash: float = 10000) -> dict:
    df = pd.read_sql(f"SELECT * FROM {ticker}", conn, index_col="date", parse_dates=["date"])
    df = df.sort_index()

    results = {"ticker": ticker}

    for name, strategy in [
        ("buy_and_hold",  strategy_buy_and_hold),
        ("ma_crossover",  strategy_ma_crossover),
        ("rsi",           strategy_rsi),
    ]:
        portfolio = strategy(df, starting_cash)
        results[name] = {
            "final_value":  round(portfolio[-1], 2),
            "total_return": total_return(portfolio),
            "max_drawdown": max_drawdown(portfolio),
            "sharpe_ratio": sharpe_ratio(portfolio),
        }

    return results


# Main

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    for ticker in TICKERS:
        r = backtest_ticker(ticker, conn)
        print(f"\n── {r['ticker']} ───────────────────────────────────")
        print(f"{'Strategy':<20} {'Final $':>10} {'Return %':>10} {'Max DD %':>10} {'Sharpe':>8}")
        print("-" * 62)
        for strategy in ["buy_and_hold", "ma_crossover", "rsi"]:
            s = r[strategy]
            print(f"{strategy:<20} {s['final_value']:>10} {s['total_return']:>9}% {s['max_drawdown']:>9}% {s['sharpe_ratio']:>8}")

    conn.close()
