import os
import sys
import logging
import sqlite3
import pandas as pd
from datetime import date

# Make project root importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.agent.buffett_advisor import get_buffett_context

# Logging
LOG_DIR  = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Buffett advisor note generator
# Translates the Buffett context into a human-readable sentence

def buffett_note(ctx: dict, adjustment: int) -> str:
    score   = ctx["buffett_score"]
    quality = ctx["data_quality"]
    trend   = ctx["earnings_trend"]
    roe     = ctx["roe"]
    pe      = ctx["pe_ratio"]

    if quality == "unavailable":
        return "Buffett: No fundamental data available — cannot advise."

    if "partial" in quality:
        note = f"Buffett: Partial data ({quality}). "
    else:
        note = "Buffett: "

    if adjustment >= 1:
        note += f"Strong fundamentals confirm this signal (score={score}/10, ROE={roe}%, earnings={trend}). High conviction."
    elif adjustment == 0:
        note += f"Fundamentals are neutral (score={score}/10). Technical signal stands on its own."
    elif adjustment == -1:
        note += f"Weak fundamentals urge caution (score={score}/10, P/E={pe}). Reduce confidence."
    else:
        note += f"Poor fundamentals argue against this trade (score={score}/10, P/E={pe}, ROE={roe}%). Proceed carefully."

    return note


# Core agent class

class StockAgent:
    def __init__(self, use_buffett: bool = False):
        # use_buffett=False → Version 1: technical only
        # use_buffett=True  → Version 2: technical + Buffett advisor
        self.use_buffett = use_buffett
        self.version     = "v2-buffett" if use_buffett else "v1-technical"

    def _technical_confidence(self, row: pd.Series) -> tuple[int, list]:
        # Score each technical signal and return total confidence + reasons
        confidence = 0
        reasons    = []

        # RSI signal — strongest weight
        rsi = row.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                confidence += 2
                reasons.append(f"RSI={rsi:.1f} (oversold)")
            elif rsi > 70:
                confidence -= 2
                reasons.append(f"RSI={rsi:.1f} (overbought)")

        # MA crossover confirmation
        ma7  = row.get("ma_7")
        ma30 = row.get("ma_30")
        if ma7 is not None and ma30 is not None:
            if ma7 > ma30:
                confidence += 1
                reasons.append("MA crossover up")
            else:
                confidence -= 1
                reasons.append("MA crossover down")

        # Bollinger Band position
        close    = row.get("Close")
        bb_upper = row.get("bb_upper")
        bb_lower = row.get("bb_lower")
        if close is not None and bb_upper is not None and bb_lower is not None:
            band_range = bb_upper - bb_lower
            if band_range > 0:
                if close <= bb_lower + (band_range * 0.1):
                    confidence += 1
                    reasons.append("price near lower Bollinger Band")
                elif close >= bb_upper - (band_range * 0.1):
                    confidence -= 1
                    reasons.append("price near upper Bollinger Band")

        return confidence, reasons

    def _buffett_adjustment(self, ticker: str) -> tuple[int, dict, str]:
        # Fetch Buffett context and return confidence adjustment + note
        ctx   = get_buffett_context(ticker)
        score = ctx["buffett_score"]

        if score is None:
            return 0, ctx, "Buffett: No score available — skipping adjustment."

        if score >= 8:
            adjustment = 1
        elif score >= 6:
            adjustment = 0
        elif score >= 4:
            adjustment = -1
        else:
            adjustment = -2

        note = buffett_note(ctx, adjustment)
        return adjustment, ctx, note

    def decide(self, ticker: str, row: pd.Series) -> dict:
        # Step 1 — technical confidence
        tech_confidence, reasons = self._technical_confidence(row)

        # Step 2 — Buffett adjustment (Version 2 only)
        buffett_adjustment = 0
        buffett_note_text  = "Buffett advisor not active."
        buffett_ctx        = None

        if self.use_buffett:
            buffett_adjustment, buffett_ctx, buffett_note_text = self._buffett_adjustment(ticker)

        # Step 3 — final decision
        final_confidence = tech_confidence + buffett_adjustment

        if final_confidence >= 2:
            action = "BUY"
        elif final_confidence <= -2:
            action = "SELL"
        else:
            action = "HOLD"

        result = {
            "ticker":               ticker,
            "action":               action,
            "technical_confidence": tech_confidence,
            "buffett_adjustment":   buffett_adjustment,
            "final_confidence":     final_confidence,
            "reasons":              reasons,
            "buffett_note":         buffett_note_text,
            "version":              self.version,
            "date":                 str(row.name) if hasattr(row, "name") else str(date.today()),
        }

        # Log the decision with full reasoning
        log.info(
            f"{ticker} [{self.version}] → {action} | "
            f"technical={tech_confidence} ({', '.join(reasons)}) | "
            f"buffett_adj={buffett_adjustment} | final={final_confidence}"
        )
        if self.use_buffett:
            log.info(f"  {buffett_note_text}")

        return result


# Run both agent versions on the latest data row for each ticker

if __name__ == "__main__":
    import yaml

    CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    DB_PATH = os.path.join(BASE_DIR, config["data"]["db_path"])
    TICKERS = config["tickers"]

    agent_v1 = StockAgent(use_buffett=False)
    agent_v2 = StockAgent(use_buffett=True)

    conn = sqlite3.connect(DB_PATH)

    print("\n" + "=" * 70)
    print("  STOCK AGENT — TODAY'S DECISIONS")
    print("=" * 70)

    for ticker in TICKERS:
        df  = pd.read_sql(f"SELECT * FROM {ticker}", conn, index_col="date", parse_dates=["date"])
        df  = df.sort_index()
        row = df.iloc[-1]  # most recent day

        print(f"\n── {ticker} ───────────────────────────────────────────")

        # Version 1 — technical only
        d1 = agent_v1.decide(ticker, row)
        print(f"  V1 Technical:  {d1['action']} (confidence={d1['final_confidence']})")
        print(f"  Signals: {', '.join(d1['reasons'])}")

        print()

        # Version 2 — technical + Buffett
        d2 = agent_v2.decide(ticker, row)
        print(f"  V2 w/Buffett:  {d2['action']} (confidence={d2['final_confidence']})")
        print(f"  Signals: {', '.join(d2['reasons'])}")
        print(f"  {d2['buffett_note']}")

    conn.close()
    print("\n" + "=" * 70)
