import json
from pathlib import Path

import pandas as pd
import requests

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
SYMBOL = "GBPUSD=X"
OUT_DIR = Path("market_candles")
HEADERS = {"User-Agent": "Mozilla/5.0"}

REQUESTS = {
    "D1": ("1d", "2y"),
    "H1": ("1h", "730d"),
    "M15": ("15m", "60d"),
    "W1": ("1wk", "5y"),
}


def download(interval, period):
    url = YAHOO_URL + requests.utils.quote(SYMBOL, safe="")
    response = requests.get(
        url,
        params={"range": period, "interval": interval, "events": "history"},
        headers=HEADERS,
        timeout=45,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise RuntimeError(str(chart["error"]))
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo returned no chart results")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes_list = result.get("indicators", {}).get("quote") or []
    if not timestamps or not quotes_list:
        raise RuntimeError("Yahoo returned no OHLC data")
    quote = quotes_list[0]
    volume = quote.get("volume")
    if volume is None:
        volume = [None] * len(timestamps)
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "Open": quote.get("open", []),
        "High": quote.get("high", []),
        "Low": quote.get("low", []),
        "Close": quote.get("close", []),
        "Volume": volume,
    })
    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last").reset_index(drop=True)

    now = pd.Timestamp.now(tz="UTC")
    durations = {
        "1d": pd.Timedelta(days=1),
        "1h": pd.Timedelta(hours=1),
        "15m": pd.Timedelta(minutes=15),
        "1wk": pd.Timedelta(days=7),
    }
    df = df.loc[df["Timestamp"] + durations[interval] <= now].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No completed candles after filtering")
    return df


def build_h4(h1):
    data = h1.copy().set_index("Timestamp")
    grouped = data.resample("4h", origin="epoch", label="left", closed="left")
    h4 = grouped.agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    h4["Count"] = grouped["Close"].count()
    h4 = h4.loc[h4["Count"] == 4].drop(columns=["Count"])
    h4 = h4.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
    now = pd.Timestamp.now(tz="UTC")
    h4 = h4.loc[h4["Timestamp"] + pd.Timedelta(hours=4) <= now]
    return h4.reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, errors = {}, {}
    for tf, (interval, period) in REQUESTS.items():
        try:
            frames[tf] = download(interval, period)
            print(f"{tf}: {len(frames[tf])} completed candles")
        except Exception as exc:
            errors[tf] = str(exc)
            print(f"ERROR {tf}: {exc}")

    if "H1" in frames:
        frames["H4"] = build_h4(frames["H1"])
        print(f"H4: {len(frames['H4'])} completed candles built from H1")

    if not frames:
        raise RuntimeError("No timeframe data downloaded successfully")

    manifest = {
        "symbol": "GBPUSD",
        "yahoo_symbol": SYMBOL,
        "source": "Yahoo Finance chart API",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "timeframes": {},
        "errors": errors,
        "note": "Completed candles only. H4 is aggregated from H1. All timestamps are UTC.",
    }
    combined = []
    for tf, frame in frames.items():
        df = frame.copy()
        df.insert(0, "Timeframe", tf)
        df.insert(1, "Symbol", "GBPUSD")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        filename = f"GBPUSD_{tf}.csv"
        df.to_csv(OUT_DIR / filename, index=False)
        manifest["timeframes"][tf] = {
            "file": filename,
            "candles": int(len(df)),
            "first_candle_utc": str(df["Timestamp"].iloc[0]) if len(df) else None,
            "last_candle_utc": str(df["Timestamp"].iloc[-1]) if len(df) else None,
        }
        combined.append(df)

    pd.concat(combined, ignore_index=True).to_csv(OUT_DIR / "GBPUSD_ALL_TIMEFRAMES.csv", index=False)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
