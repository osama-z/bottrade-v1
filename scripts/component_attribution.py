"""component_attribution.py — does each AI component earn its weight?

Joins the audit trail (signals → trades on correlation_id) and reports,
per component score (llm / ml / sentiment / combined):

- Pearson correlation of the score with realized trade PnL
- directional hit rate: how often the score's SIGN agreed with the
  trade's outcome (win = pnl > 0, per risk/metrics.py)

Run after a stretch of paper trading:

    python scripts/component_attribution.py            # configured DB
    python scripts/component_attribution.py path/to.db

Interpretation: a component whose correlation ≈ 0 and hit rate ≈ 50% is
adding cost and noise, not alpha — set its combiner weight to zero (or
remove it) before spending money on a better model for it.
"""

import sqlite3
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from risk import metrics

COMPONENTS = ("llm_score", "ml_score", "sentiment_score", "combined_score")


def load_joined(db_path: str) -> pd.DataFrame:
    """Closed trades joined to the signal that produced them."""
    query = """
        SELECT s.llm_score, s.ml_score, s.sentiment_score, s.combined_score,
               s.decision, t.pnl, t.symbol
        FROM trades t
        JOIN signals s
          ON s.correlation_id = t.correlation_id
         AND s.symbol = t.symbol
        WHERE t.status = 'closed'
          AND t.pnl IS NOT NULL
          AND t.correlation_id != ''
          AND s.correlation_id != ''
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-component correlation and directional hit rate vs PnL."""
    rows = []
    wins = df["pnl"].map(metrics.is_win)
    for comp in COMPONENTS:
        scores = df[comp].fillna(0.0)
        corr = scores.corr(df["pnl"]) if scores.nunique() > 1 else float("nan")
        # sign agreement: positive score should mean winning trade
        decided = scores != 0
        if decided.any():
            hit_rate = float(((scores > 0) == wins)[decided].mean())
        else:
            hit_rate = float("nan")
        rows.append({
            "component": comp,
            "n_trades": int(len(df)),
            "n_nonzero": int(decided.sum()),
            "pnl_correlation": round(float(corr), 4) if pd.notna(corr) else None,
            "directional_hit_rate": round(hit_rate, 4) if pd.notna(hit_rate) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        from config.settings import settings
        db_path = settings.database_path

    df = load_joined(db_path)
    if df.empty:
        print(
            "No joined signal→trade rows yet. Correlation IDs are written "
            "for every cycle since PR #5 — run the bot, then re-run this."
        )
        return

    print(f"\nComponent attribution over {len(df)} closed trades ({db_path})\n")
    report = attribution(df)
    print(report.to_string(index=False))
    print(
        "\nReading guide: pnl_correlation near 0 and hit rate near 0.5 "
        "means the component is NOT contributing — zero its weight before "
        "upgrading its model. Win = pnl > 0 (risk/metrics.py)."
    )


if __name__ == "__main__":
    main()
