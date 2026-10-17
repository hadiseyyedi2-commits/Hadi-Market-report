import os
import json
import time
import base64
import hashlib
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

MAX_SIGNAL_AGE = 5
MAX_STORED_SIGNALS = 500

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
GITHUB_BRANCH = os.getenv("GITHUB_REF_NAME", "main")

STATE_PATH = "state/sent_signals.json"

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "EURCHF": "EURCHF=X",
    "EURCAD": "EURCAD=X",
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "GBPJPY": "GBPJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPCAD": "GBPCAD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",
    "AUDJPY": "AUDJPY=X",
    "AUDCAD": "AUDCAD=X",
    "AUDNZD": "AUDNZD=X",
    "NZDJPY": "NZDJPY=X",
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
}

TIMEFRAMES = {
    "D1": ("1d", "2y"),
    "H4": ("4h", "730d"),
    "H1": ("1h", "730d"),
    "M15": ("15m", "60d"),
    "M5": ("5m", "60d"),
}


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Telegram credentials are missing.")
        return False

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_BOT_TOKEN
        + "/sendMessage"
    )

    chunks = []

    while len(message) > 3900:
        split_at = message.rfind("\n", 0, 3900)

        if split_at <= 0:
            split_at = 3900

        chunks.append(message[:split_at])
        message = message[split_at:].lstrip()

    if message:
        chunks.append(message)

    for chunk in chunks:
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk
                },
                timeout=30
            )

            if response.status_code != 200:
                print(
                    "Telegram error:",
                    response.status_code,
                    response.text[:500]
                )
                return False

            result = response.json()

            if not result.get("ok", False):
                print("Telegram rejected message:", result)
                return False

        except requests.RequestException as exc:
            print("Telegram request failed:", exc)
            return False

    return True


# ============================================================
# GITHUB STATE STORAGE
# ============================================================

def github_url(path):
    return (
        "https://api.github.com/repos/"
        + GITHUB_REPOSITORY
        + "/"
        + path.lstrip("/")
    )


def github_headers():
    return {
        "Authorization": "Bearer " + GITHUB_TOKEN,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }


def load_state():
    if GITHUB_TOKEN and GITHUB_REPOSITORY:
        try:
            response = requests.get(
                github_url("contents/" + STATE_PATH),
                headers=github_headers(),
                params={"ref": GITHUB_BRANCH},
                timeout=30
            )

            if response.status_code == 200:
                payload = response.json()

                content = base64.b64decode(
                    payload["content"]
                ).decode("utf-8")

                state = json.loads(content)

                if isinstance(state, dict):
                    return state

            elif response.status_code != 404:
                print(
                    "GitHub state read error:",
                    response.status_code,
                    response.text[:300]
                )

        except Exception as exc:
            print("GitHub state read failed:", exc)

    try:
        if os.path.exists(STATE_PATH):
            with open(
                STATE_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                state = json.load(file)

                if isinstance(state, dict):
                    return state

    except Exception as exc:
        print("Local state read failed:", exc)

    return {}


def save_state(state):
    os.makedirs(
        os.path.dirname(STATE_PATH),
        exist_ok=True
    )

    with open(
        STATE_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )

    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("WARNING: GitHub persistence is unavailable.")
        return False

    url = github_url("contents/" + STATE_PATH)

    try:
        existing = requests.get(
            url,
            headers=github_headers(),
            params={"ref": GITHUB_BRANCH},
            timeout=30
        )

        payload = {
            "message": "Update Level Criterion signal history",
            "content": base64.b64encode(
                json.dumps(
                    state,
                    ensure_ascii=False,
                    indent=2
                ).encode("utf-8")
            ).decode("utf-8"),
            "branch": GITHUB_BRANCH
        }

        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]

        elif existing.status_code != 404:
            print(
                "GitHub state check failed:",
                existing.status_code,
                existing.text[:300]
            )
            return False

        response = requests.put(
            url,
            headers=github_headers(),
            json=payload,
            timeout=30
        )

        if response.status_code in (200, 201):
            print("Signal history saved to GitHub.")
            return True

        print(
            "GitHub state save failed:",
            response.status_code,
            response.text[:500]
        )

    except Exception as exc:
        print("GitHub state save exception:", exc)

    return False


# ============================================================
# MARKET DATA DOWNLOAD
# ============================================================

def download_data(symbol, interval, period):
    url = YAHOO_URL + requests.utils.quote(
        symbol,
        safe=""
    )

    params = {
        "range": period,
        "interval": interval,
        "events": "history"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=45
        )

        response.raise_for_status()

        payload = response.json()
        chart = payload.get("chart", {})

        if chart.get("error"):
            print(
                "Yahoo error:",
                symbol,
                interval,
                chart["error"]
            )
            return None

        results = chart.get("result")

        if not results:
            return None

        result = results[0]
        timestamps = result.get("timestamp", [])

        if not timestamps:
            return None

        indicators = result.get("indicators", {})
        quote_list = indicators.get("quote", [])

        if not quote_list:
            return None

        quotes = quote_list[0]

        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(
                timestamps,
                unit="s",
                utc=True
            ),
            "Open": quotes.get("open", []),
            "High": quotes.get("high", []),
            "Low": quotes.get("low", []),
            "Close": quotes.get("close", [])
        })

        df = df.dropna(
            subset=[
                "Timestamp",
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        df = df.sort_values("Timestamp")
        df = df.drop_duplicates(
            subset=["Timestamp"],
            keep="last"
        ).reset_index(drop=True)

        if df.empty:
            return None

        now = pd.Timestamp.now(tz="UTC")

        interval_minutes = {
            "1d": 1440,
            "1h": 60,
            "15m": 15,
            "5m": 5
        }.get(interval)

        if interval_minutes:
            candle_end = (
                df["Timestamp"]
                + pd.to_timedelta(
                    interval_minutes,
                    unit="m"
                )
            )

            df = df.loc[
                candle_end <= now
            ].copy()

        df = df.reset_index(drop=True)

        if len(df) < 50:
            print(
                "Insufficient data:",
                symbol,
                interval,
                len(df)
            )
            return None

        return df

    except Exception as exc:
        print(
            "Download failed:",
            symbol,
            interval,
            str(exc)[:300]
        )
        return None


# ============================================================
# BUILD H4 FROM H1
# ============================================================

def build_h4_from_h1(h1):
    if h1 is None or h1.empty:
        return None

    data = h1.copy().set_index("Timestamp")

    grouped = data.resample(
        "4h",
        origin="epoch",
        label="left",
        closed="left"
    )

    h4 = grouped.agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    counts = grouped["Close"].count()
    h4["Count"] = counts

    h4 = h4.loc[
        h4["Count"] == 4
    ].copy()

    h4 = h4.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    h4 = h4.drop(columns=["Count"]).reset_index()

    now = pd.Timestamp.now(tz="UTC")

    h4 = h4.loc[
        h4["Timestamp"] + pd.Timedelta(hours=4)
        <= now
    ].copy()

    h4 = h4.reset_index(drop=True)

    if len(h4) < 50:
        return None

    return h4


# ============================================================
# MACD 12,26,9
# ============================================================

def calculate_macd(df):
    result = df.copy()

    ema_fast = result["Close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    ema_slow = result["Close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    result["MACD"] = ema_fast - ema_slow

    result["MACD_Signal"] = result["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    result["MACD_Hist"] = (
        result["MACD"] - result["MACD_Signal"]
    )

    return result


# ============================================================
# HISTOGRAM PHASES
# ============================================================

def get_macd_phases(df):
    values = df["MACD_Hist"].to_numpy(dtype=float)

    signs = np.sign(values).astype(float)
    signs[signs == 0] = np.nan

    sign_series = pd.Series(signs).ffill().bfill()

    if sign_series.isna().all():
        return []

    signs = sign_series.to_numpy(dtype=float)

    phases = []
    start = 0
    current_sign = int(signs[0])

    for i in range(1, len(signs)):
        sign_now = int(signs[i])

        if sign_now != current_sign:
            phases.append({
                "sign": current_sign,
                "start": start,
                "end": i - 1
            })

            start = i
            current_sign = sign_now

    phases.append({
        "sign": current_sign,
        "start": start,
        "end": len(signs) - 1
    })

    return phases


# ============================================================
# PRICE EXTREMES
# ============================================================

def get_extreme(df, start, end, kind):
    section = df.iloc[start:end + 1]

    if section.empty:
        return None

    if kind == "LOW":
        idx = section["Low"].idxmin()
        price = float(df.loc[idx, "Low"])
    else:
        idx = section["High"].idxmax()
        price = float(df.loc[idx, "High"])

    row = df.loc[idx]

    return {
        "index": int(idx),
        "time": row["Timestamp"],
        "price": price,
        "macd": float(row["MACD_Hist"]),
        "kind": kind
    }


# ============================================================
# HISTOGRAM EXTREMES ACROSS THE ENTIRE PHASE
# ============================================================

def get_histogram_extreme(df, start, end, kind):
    section = df.iloc[start:end + 1]

    if section.empty:
        return None

    if kind == "LOW":
        idx = section["MACD_Hist"].idxmin()
    else:
        idx = section["MACD_Hist"].idxmax()

    row = df.loc[idx]

    return {
        "index": int(idx),
        "time": row["Timestamp"],
        "value": float(row["MACD_Hist"])
    }


# ============================================================
# SIGNAL ID
# ============================================================

def make_signal_id(
    symbol,
    timeframe,
    direction,
    points,
    trigger_time
):
    parts = [
        symbol,
        timeframe,
        direction,
        str(trigger_time)
    ]

    for name, point in points:
        parts.append(name)
        parts.append(str(point["time"]))

    return hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()


# ============================================================
# LEVEL CRITERION DETECTION - TLM-04
# ============================================================

def detect_level_criterion(df, symbol, timeframe):
    """
    TLM-04 uses the MACD histogram (MACD - Signal), settings 12,26,9.

    SELL:
      N1 -> P1 -> N2 -> P2

      1. Price Low(N2) < Price Low(N1)
      2. Minimum histogram over the entire N2 phase
         > minimum histogram over the entire N1 phase
      3. Price High(P2) < Price High(P1)
      4. Maximum histogram over the entire P2 phase
         > maximum histogram over the entire P1 phase

    BUY:
      P1 -> N1 -> P2 -> N2

      1. Price High(P2) > Price High(P1)
      2. Maximum histogram over the entire P2 phase
         < maximum histogram over the entire P1 phase
      3. Price Low(N2) > Price Low(N1)
      4. Minimum histogram over the entire N2 phase
         < minimum histogram over the entire N1 phase

    All four phases must be consecutive and closed.
    A qualifying pattern is reported only if its fourth phase
    has ended within the last MAX_SIGNAL_AGE closed candles.

    No separate histogram breakout is added because it would
    contradict the whole-phase hidden-divergence condition.
    """

    if df is None or len(df) < 50:
        return []

    data = calculate_macd(df)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "MACD_Hist",
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ).reset_index(drop=True)

    if len(data) < 50:
        return []

    phases = get_macd_phases(data)

    if len(phases) < 4:
        return []

    last_closed_index = len(data) - 1
    found = []

    for i in range(len(phases) - 3):
        phase1 = phases[i]
        phase2 = phases[i + 1]
        phase3 = phases[i + 2]
        phase4 = phases[i + 3]

        signs = [
            phase1["sign"],
            phase2["sign"],
            phase3["sign"],
            phase4["sign"]
        ]

        direction = None
        points = []

        # ----------------------------------------------------
        # SELL: N1 -> P1 -> N2 -> P2
        # ----------------------------------------------------

        if signs == [-1, 1, -1, 1]:

            N1 = get_extreme(
                data,
                phase1["start"],
                phase1["end"],
                "LOW"
            )

            P1 = get_extreme(
                data,
                phase2["start"],
                phase2["end"],
                "HIGH"
            )

            N2 = get_extreme(
                data,
                phase3["start"],
                phase3["end"],
                "LOW"
            )

            P2 = get_extreme(
                data,
                phase4["start"],
                phase4["end"],
                "HIGH"
            )

            n1_hist_low = get_histogram_extreme(
                data,
                phase1["start"],
                phase1["end"],
                "LOW"
            )

            p1_hist_high = get_histogram_extreme(
                data,
                phase2["start"],
                phase2["end"],
                "HIGH"
            )

            n2_hist_low = get_histogram_extreme(
                data,
                phase3["start"],
                phase3["end"],
                "LOW"
            )

            p2_hist_high = get_histogram_extreme(
                data,
                phase4["start"],
                phase4["end"],
                "HIGH"
            )

            if any(
                p is None
                for p in [
                    N1, P1, N2, P2,
                    n1_hist_low, p1_hist_high,
                    n2_hist_low, p2_hist_high
                ]
            ):
                continue

            valid = (
                N2["price"] < N1["price"]
                and n2_hist_low["value"] > n1_hist_low["value"]
                and P2["price"] < P1["price"]
                and p2_hist_high["value"] > p1_hist_high["value"]
            )

            if not valid:
                continue

            # Fourth phase must have completed.
            trigger_index = phase4["end"]

            direction = "SELL"

            points = [
                ("N1", N1),
                ("P1", P1),
                ("N2", N2),
                ("P2", P2)
            ]

        # ----------------------------------------------------
        # BUY: P1 -> N1 -> P2 -> N2
        # ----------------------------------------------------

        elif signs == [1, -1, 1, -1]:

            P1 = get_extreme(
                data,
                phase1["start"],
                phase1["end"],
                "HIGH"
            )

            N1 = get_extreme(
                data,
                phase2["start"],
                phase2["end"],
                "LOW"
            )

            P2 = get_extreme(
                data,
                phase3["start"],
                phase3["end"],
                "HIGH"
            )

            N2 = get_extreme(
                data,
                phase4["start"],
                phase4["end"],
                "LOW"
            )

            p1_hist_high = get_histogram_extreme(
                data,
                phase1["start"],
                phase1["end"],
                "HIGH"
            )

            n1_hist_low = get_histogram_extreme(
                data,
                phase2["start"],
                phase2["end"],
                "LOW"
            )

            p2_hist_high = get_histogram_extreme(
                data,
                phase3["start"],
                phase3["end"],
                "HIGH"
            )

            n2_hist_low = get_histogram_extreme(
                data,
                phase4["start"],
                phase4["end"],
                "LOW"
            )

            if any(
                p is None
                for p in [
                    P1, N1, P2, N2,
                    p1_hist_high, n1_hist_low,
                    p2_hist_high, n2_hist_low
                ]
            ):
                continue

            valid = (
                P2["price"] > P1["price"]
                and p2_hist_high["value"] < p1_hist_high["value"]
                and N2["price"] > N1["price"]
                and n2_hist_low["value"] < n1_hist_low["value"]
            )

            if not valid:
                continue

            # Fourth phase must have completed.
            trigger_index = phase4["end"]

            direction = "BUY"

            points = [
                ("P1", P1),
                ("N1", N1),
                ("P2", P2),
                ("N2", N2)
            ]

        else:
            continue

        # ----------------------------------------------------
        # FRESHNESS CHECK
        # ----------------------------------------------------

        age = last_closed_index - trigger_index

        if age < 0 or age > MAX_SIGNAL_AGE:
            continue

        trigger_time = data.loc[
            trigger_index,
            "Timestamp"
        ]

        signal_id = make_signal_id(
            symbol,
            timeframe,
            direction,
            points,
            trigger_time
        )

        found.append({
            "id": signal_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "age": age,
            "points": points,
            "final_time": trigger_time,
            "trigger_time": trigger_time
        })

    found.sort(
        key=lambda item: item["final_time"],
        reverse=True
    )

    # Keep the newest signal per direction for this symbol/timeframe.
    unique = []
    seen_directions = set()

    for signal in found:
        if signal["direction"] in seen_directions:
            continue

        seen_directions.add(signal["direction"])
        unique.append(signal)

    return unique


# ============================================================
# REPORT FORMAT
# ============================================================

def format_timestamp(value):
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def format_price(value):
    if value == 0:
        return "0"

    if abs(value) >= 100:
        return f"{value:.3f}"

    if abs(value) >= 1:
        return f"{value:.5f}"

    return f"{value:.6f}"


def format_signal(signal):
    marker = (
        "🟢 BUY"
        if signal["direction"] == "BUY"
        else "🔻 SELL"
    )

    lines = [
        "★★ LEVEL CRITERION / TLM-04 ★★",
        "",
        f"{marker} | {signal['symbol']}",
        f"Timeframe: {signal['timeframe']}",
        f"Confirmation candle age: {signal['age']} candle(s)",
        f"Confirmation time: {format_timestamp(signal['trigger_time'])}",
        ""
    ]

    for name, point in signal["points"]:
        lines.append(
            f"{name}: "
            f"Price={format_price(point['price'])} | "
            f"Histogram={point['macd']:.8f} | "
            f"Time={format_timestamp(point['time'])}"
        )

    lines.extend([
        "",
        f"MACD Histogram: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}"
    ])

    return "\n".join(lines)


# ============================================================
# MAIN SCANNER
# ============================================================

def main():
    print("=" * 60)
    print("HADI MARKET REPORT")
    print("LEVEL CRITERION / TLM-04")
    print(
        f"MACD Histogram: "
        f"{MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}"
    )
    print(
        f"Maximum signal age: "
        f"{MAX_SIGNAL_AGE} closed candles"
    )
    print("Timeframes: D1, H4, H1, M15, M5")
    print("=" * 60)

    sent_state = load_state()
    discovered = []

    for symbol_name, yahoo_symbol in SYMBOLS.items():
        print("\nScanning:", symbol_name)

        data_by_timeframe = {}

        # ----------------------------------------------------
        # D1
        # ----------------------------------------------------

        d1 = download_data(
            yahoo_symbol,
            "1d",
            "2y"
        )

        if d1 is not None:
            data_by_timeframe["D1"] = d1

        time.sleep(0.2)

        # ----------------------------------------------------
        # H1 and H4
        # ----------------------------------------------------

        h1 = download_data(
            yahoo_symbol,
            "1h",
            "730d"
        )

        if h1 is not None:
            data_by_timeframe["H1"] = h1

            h4 = build_h4_from_h1(h1)

            if h4 is not None:
                data_by_timeframe["H4"] = h4

        time.sleep(0.2)

        # ----------------------------------------------------
        # M15
        # ----------------------------------------------------

        m15 = download_data(
            yahoo_symbol,
            "15m",
            "60d"
        )

        if m15 is not None:
            data_by_timeframe["M15"] = m15

        time.sleep(0.2)

        # ----------------------------------------------------
        # M5
        # ----------------------------------------------------

        m5 = download_data(
            yahoo_symbol,
            "5m",
            "60d"
        )

        if m5 is not None:
            data_by_timeframe["M5"] = m5

        time.sleep(0.2)

        # ----------------------------------------------------
        # SCAN ALL REQUESTED TIMEFRAMES
        # ----------------------------------------------------

        for timeframe in TIMEFRAMES:
            df = data_by_timeframe.get(timeframe)

            if df is None:
                print(
                    "No data:",
                    symbol_name,
                    timeframe
                )
                continue

            signals = detect_level_criterion(
                df,
                symbol_name,
                timeframe
            )

            for signal in signals:
                if signal["id"] in sent_state:
                    continue

                discovered.append(signal)

                print(
                    "NEW:",
                    signal["direction"],
                    symbol_name,
                    timeframe,
                    "age:",
                    signal["age"]
                )

    if not discovered:
        print("\nNo new Level Criterion signals found.")
        return

    discovered.sort(
        key=lambda item: item["final_time"],
        reverse=True
    )

    report = "\n\n--------------------\n\n".join(
        format_signal(signal)
        for signal in discovered
    )

    print("\nReport:")
    print(report)

    if not send_telegram(report):
        print(
            "Telegram delivery failed. "
            "Signals were not marked as sent."
        )
        return

    now = datetime.now(timezone.utc).isoformat()

    for signal in discovered:
        sent_state[signal["id"]] = {
            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],
            "direction": signal["direction"],
            "sent_at": now,
            "final_time": str(signal["final_time"]),
            "trigger_time": str(signal["trigger_time"])
        }

    if len(sent_state) > MAX_STORED_SIGNALS:
        ordered = sorted(
            sent_state.items(),
            key=lambda item: item[1].get("sent_at", ""),
            reverse=True
        )

        sent_state = dict(
            ordered[:MAX_STORED_SIGNALS]
        )

    if not save_state(sent_state):
        print(
            "WARNING: Telegram was sent, but GitHub state "
            "persistence failed. Duplicate alerts may occur."
        )

    print("\nTelegram report sent successfully.")


if __name__ == "__main__":
    main()