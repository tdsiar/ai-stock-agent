import yfinance as yf

# This module is the data foundation for the "Ghost of Buffett" agent.
# It collects fundamental data and scores each ticker using Buffett's
# core investing principles. No AI calls, no decisions — data and scores only.
# In Stage 5, get_buffett_context() will feed directly into the agent's
# decision-making prompt.


# Scoring helpers
# Each function returns a score between 0 and 2.5 (four metrics = 10 total)

def score_pe_ratio(pe):
    # Buffett prefers P/E below 15 (great value), tolerates up to 25
    # Returns None if data is missing — never silently default
    if pe is None:
        return None
    if pe < 15:
        return 2.5
    elif pe < 25:
        return 1.5
    elif pe < 40:
        return 0.5
    else:
        return 0.0


def score_roe(roe):
    # Buffett loves ROE above 15% — it means the company earns well on shareholder money
    if roe is None:
        return None
    roe_pct = roe * 100
    if roe_pct >= 20:
        return 2.5
    elif roe_pct >= 15:
        return 2.0
    elif roe_pct >= 10:
        return 1.0
    else:
        return 0.0


def score_debt_to_equity(de):
    # Buffett avoids heavily indebted companies — lower is better
    if de is None:
        return None
    if de < 0.5:
        return 2.5
    elif de < 1.0:
        return 1.5
    elif de < 2.0:
        return 0.5
    else:
        return 0.0


def score_earnings_trend(trend):
    # Consistent earnings growth is a Buffett hallmark
    if trend == "growing":
        return 2.5
    elif trend == "volatile":
        return 1.0
    elif trend == "declining":
        return 0.0
    else:
        return None  # unknown trend — do not score


# Earnings trend detection
# Looks at the last 4 quarters of earnings and determines direction

def detect_earnings_trend(ticker_obj):
    try:
        # Use income_stmt — yfinance's current way to get net income by quarter
        stmt = ticker_obj.quarterly_income_stmt
        if stmt is None or stmt.empty:
            return "unknown"

        if "Net Income" not in stmt.index:
            return "unknown"

        # Get the 4 most recent quarters (columns are sorted newest first)
        recent = stmt.loc["Net Income"].dropna().iloc[:4].tolist()
        if len(recent) < 4:
            return "unknown"

        # Reverse so we go oldest to newest for comparison
        recent = list(reversed(recent))

        ups   = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        downs = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])

        if ups >= 3:
            return "growing"
        elif downs >= 3:
            return "declining"
        else:
            return "volatile"

    except Exception:
        return "unknown"


# Price change calculations

def get_price_changes(ticker_obj):
    try:
        hist = ticker_obj.history(period="35d")
        if hist.empty or len(hist) < 2:
            return None, None

        latest = hist["Close"].iloc[-1]
        prev_day = hist["Close"].iloc[-2]
        month_ago = hist["Close"].iloc[0]

        change_1d = round((latest - prev_day) / prev_day * 100, 2)
        change_30d = round((latest - month_ago) / month_ago * 100, 2)
        return change_1d, change_30d

    except Exception:
        return None, None


# Main context builder
# This is what the Stage 5 agent will call to get the full Buffett picture

def get_buffett_context(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    pe_ratio       = info.get("trailingPE")
    roe            = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None:
        debt_to_equity = debt_to_equity / 100  # yfinance returns as percentage

    earnings_trend = detect_earnings_trend(t)
    change_1d, change_30d = get_price_changes(t)

    # Score each metric — None means data was unavailable
    scores = {
        "pe_ratio":       score_pe_ratio(pe_ratio),
        "roe":            score_roe(roe),
        "debt_to_equity": score_debt_to_equity(debt_to_equity),
        "earnings_trend": score_earnings_trend(earnings_trend),
    }

    # Track which metrics we could and could not read
    missing_metrics  = [k for k, v in scores.items() if v is None]
    available_scores = [v for v in scores.values() if v is not None]

    # Scale the score based on only what we actually know
    # e.g. if only 3 of 4 metrics are available, score out of 7.5 then scale to 10
    if available_scores:
        max_possible  = len(available_scores) * 2.5
        buffett_score = round(sum(available_scores) / max_possible * 10, 2)
    else:
        buffett_score = None

    # Data quality flag — Stage 5 agent will use this to caveat its response
    if not missing_metrics:
        data_quality = "complete"
    elif len(missing_metrics) == len(scores):
        data_quality = "unavailable"
    else:
        data_quality = f"partial — missing: {', '.join(missing_metrics)}"

    context = {
        "ticker":           ticker,
        "price_change_1d":  change_1d,
        "price_change_30d": change_30d,
        "pe_ratio":         pe_ratio,
        "roe":              round(roe * 100, 2) if roe is not None else None,
        "debt_to_equity":   round(debt_to_equity, 2) if debt_to_equity is not None else None,
        "earnings_trend":   earnings_trend,
        "buffett_score":    buffett_score,
        "data_quality":     data_quality,    # "complete", "partial — missing: x", or "unavailable"
        "missing_metrics":  missing_metrics, # list of fields the agent could not read
        "trigger_event":    None,            # Will be populated in Stage 5
    }

    return context


# Trigger conditions — defined here as placeholders, activated in Stage 5
# These will detect events that cause the agent to speak up unprompted

# def trigger_single_day_drop(context):
#     # Fire if price dropped more than 5% in a single day
#     return context["price_change_1d"] is not None and context["price_change_1d"] < -5

# def trigger_sector_selloff(contexts):
#     # Fire if 3 or more tickers dropped more than 3% on the same day
#     drops = [c for c in contexts if c["price_change_1d"] is not None and c["price_change_1d"] < -3]
#     return len(drops) >= 3

# def trigger_earnings_report(ticker):
#     # Fire if an earnings report was detected for this ticker today
#     pass

# def trigger_manual_query(query):
#     # Fire when the user asks a question directly
#     pass


# Quick test — run this file directly to see output for all tickers
if __name__ == "__main__":
    import yaml
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(BASE_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)

    for ticker in config["tickers"]:
        print(f"\nFetching {ticker}...")
        ctx = get_buffett_context(ticker)
        for k, v in ctx.items():
            print(f"  {k}: {v}")
