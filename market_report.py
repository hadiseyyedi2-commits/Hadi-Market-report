import os
import time
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ============================================================
# Hadi Market Reports
# Multi-Timeframe: D1, H4, H1, M15
# MACD: 12, 26, 9
# Four-phase Hidden Divergence
# ============================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X",
    "EURNZD=X", "EURCAD=X", "GBPJPY=X", "GBPCHF=X",
    "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDNZD=X", "AUDCAD=X", "NZDJPY=X",
    "XAUUSD=X", "XAGUSD=X"
]

TIMEFRAMES = [
    ("D1", "1d", "1y"),
    ("H1", "1h", "60d"),
    ("M15", "15m", "60d")
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing."
        )

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    chunks = []
    current = ""

    for line in message.splitlines():
        if len(current) + len(line) + 1 > 3900:
            chunks.append(current)
            current = ""

        current += ("\n" if current else "") + line

    if current:
        chunks.append(current)

    for chunk in chunks:
        response = SESSION.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": chunk
            },
            timeout=30
        )

        response.raise_for_status()

        if not response.json().get("ok"):
            raise RuntimeError("Telegram rejected the message.")

        time.sleep(0.4)


# ============================================================
# MARKET DATA
# ============================================================

def fetch_data(symbol, interval, period):
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + symbol
    )

    response = SESSION.get(
        url,
        params={
            "interval": interval,
            "range": period
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()
    result = data.get("chart", {}).get("result")

    if not result:
        raise ValueError("No market data returned.")

    item = result[0]
    timestamps = item.get("timestamp", [])
    quote = item["indicators"]["quote"][0]

    rows = []

    for i, timestamp in enumerate(timestamps):
        try:
            o = quote["open"][i]
            h = quote["high"][i]
            l = quote["low"][i]
            c = quote["close"][i]

            if None in (o, h, l, c):
                continue

            rows.append({
                "time": timestamp,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c)
            })

        except (IndexError, TypeError, ValueError):
            continue

    return rows


def build_h4(hourly_rows):
    """Aggregate hourly candles into UTC four-hour candles."""

    buckets = {}

    for row in hourly_rows:
        bucket = (row["time"] // 14400) * 14400

        if bucket not in buckets:
            buckets[bucket] = {
                "time": bucket,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "_last": row["time"]
            }
        else:
            candle = buckets[bucket]

            candle["high"] = max(
                candle["high"], row["high"]
            )

            candle["low"] = min(
                candle["low"], row["low"]
            )

            if row["time"] < candle["time"]:
                candle["open"] = row["open"]

            if row["time"] >= candle["_last"]:
                candle["close"] = row["close"]
                candle["_last"] = row["time"]

    output = []

    for candle in sorted(
        buckets.values(),
        key=lambda item: item["time"]
    ):
        if candle["_last"] - candle["time"] >= 3 * 3600:
            candle.pop("_last", None)
            output.append(candle)

    return output


# ============================================================
# MACD 12, 26, 9
# ============================================================

def ema(values, period):
    output = [None] * len(values)

    alpha = 2.0 / (period + 1)
    state = None

    for i, value in enumerate(values):
        if value is None:
            continue

        if state is None:
            state = value
        else:
            state = alpha * value + (1 - alpha) * state

        output[i] = state

    return output


def calculate_macd(closes):
    fast = ema(closes, 12)
    slow = ema(closes, 26)

    main = []

    for i in range(len(closes)):
        if fast[i] is None or slow[i] is None:
            main.append(None)
        else:
            main.append(fast[i] - slow[i])

    signal = ema(main, 9)

    histogram = [
        (
            main[i] - signal[i]
            if main[i] is not None and signal[i] is not None
            else None
        )
        for i in range(len(main))
    ]

    return main, signal, histogram


# ============================================================
# FOUR-PHASE HIDDEN DIVERGENCE
# ============================================================

def get_macd_phases(main):
    valid = [
        (i, value)
        for i, value in enumerate(main)
        if value is not None
    ]

    if not valid:
        return []

    phases = []

    first_i, first_value = valid[0]
    sign = 1 if first_value >= 0 else -1
    points = [(first_i, first_value)]

    for i, value in valid[1:]:
        current_sign = 1 if value >= 0 else -1

        if current_sign != sign:
            phases.append({
                "sign": sign,
                "points": points
            })

            sign = current_sign
            points = [(i, value)]

        else:
            points.append((i, value))

    phases.append({
        "sign": sign,
        "points": points
    })

    return phases


def detect_hidden_divergence(rows):
    if len(rows) < 60:
        return None

    closes = [row["close"] for row in rows]

    main, signal, histogram = calculate_macd(closes)
    phases = get_macd_phases(main)

    if len(phases) < 4:
        return None

    # BUY:
    # P1 -> N1 -> P2 -> N2
    # Price Low(N2) > Price Low(N1)
    # MACD Low(N2) < MACD Low(N1)

    # SELL:
    # N1 -> P1 -> N2 -> P2
    # Price High(P2) < Price High(P1)
    # MACD High(P2) > MACD High(P1)

    for j in range(3, len(phases)):
        p1, p2, p3, p4 = phases[j - 3:j + 1]

        signs = (
            p1["sign"],
            p2["sign"],
            p3["sign"],
            p4["sign"]
        )

        # Bullish hidden divergence
        if signs == (1, -1, 1, -1):

            n1 = min(
                p2["points"],
                key=lambda point: rows[point[0]]["low"]
            )

            n2 = min(
                p4["points"],
                key=lambda point: rows[point[0]]["low"]
            )

            i1, macd1 = n1
            i2, macd2 = n2

            if (
                rows[i2]["low"] > rows[i1]["low"]
                and macd2 < macd1
            ):
                return {
                    "direction": "BUY",
                    "description": (
                        "هیدن صعودی چهار‌فازی: "
                        "کف قیمت دوم بالاتر و "
                        "کف MACD دوم پایین‌تر است."
                    )
                }

        # Bearish hidden divergence
        if signs == (-1, 1, -1, 1):

            p1_high = max(
                p2["points"],
                key=lambda point: rows[point[0]]["high"]
            )

            p2_high = max(
                p4["points"],
                key=lambda point: rows[point[0]]["high"]
            )

            i1, macd1 = p1_high
            i2, macd2 = p2_high

            if (
                rows[i2]["high"] < rows[i1]["high"]
                and macd2 > macd1
            ):
                return {
                    "direction": "SELL",
                    "description": (
                        "هیدن نزولی چهار‌فازی: "
                        "سقف قیمت دوم پایین‌تر و "
                        "سقف MACD دوم بالاتر است."
                    )
                }

    return None


# ============================================================
# MARKET STRUCTURE AND SWINGS
# ============================================================

def find_swings(rows, left=3, right=3):
    highs = []
    lows = []

    for i in range(left, len(rows) - right):
        high = rows[i]["high"]
        low = rows[i]["low"]

        if (
            all(
                high > rows[k]["high"]
                for k in range(i - left, i)
            )
            and
            all(
                high >= rows[k]["high"]
                for k in range(i + 1, i + right + 1)
            )
        ):
            highs.append((i, high))

        if (
            all(
                low < rows[k]["low"]
                for k in range(i - left, i)
            )
            and
            all(
                low <= rows[k]["low"]
                for k in range(i + 1, i + right + 1)
            )
        ):
            lows.append((i, low))

    return highs, lows


def analyze_structure(rows):
    highs, lows = find_swings(rows)
    parts = []

    if len(highs) >= 2:
        previous = highs[-2][1]
        current = highs[-1][1]

        if current > previous:
            parts.append("سقف‌ها HH")
        elif current < previous:
            parts.append("سقف‌ها LH")
        else:
            parts.append("سقف‌ها برابر")

    if len(lows) >= 2:
        previous = lows[-2][1]
        current = lows[-1][1]

        if current > previous:
            parts.append("کف‌ها HL")
        elif current < previous:
            parts.append("کف‌ها LL")
        else:
            parts.append("کف‌ها برابر")

    if not parts:
        return "ساختار بازار: داده کافی نیست"

    return "ساختار بازار: " + " | ".join(parts)


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def find_fvg(rows, lookback=100):
    zones = []

    start = max(2, len(rows) - lookback)

    for i in range(start, len(rows)):
        first = rows[i - 2]
        current = rows[i]

        if current["low"] > first["high"]:
            zones.append({
                "type": "Bullish",
                "low": first["high"],
                "high": current["low"],
                "index": i
            })

        elif current["high"] < first["low"]:
            zones.append({
                "type": "Bearish",
                "low": current["high"],
                "high": first["low"],
                "index": i
            })

    return zones


# ============================================================
# REPORT FORMAT
# ============================================================

def price(value):
    if value >= 1000:
        return f"{value:.2f}"

    if value >= 1:
        return f"{value:.5f}"

    return f"{value:.5f}"


def analyze_timeframe(tf, rows):
    if len(rows) < 60:
        return f"【{tf}】 داده کافی نیست."

    last = rows[-1]
    previous = rows[-2]

    change = last["close"] - previous["close"]

    if change > 0:
        direction = "صعودی"
    elif change < 0:
        direction = "نزولی"
    else:
        direction = "بدون تغییر"

    output = [
        f"【{tf}】",
        f"آخرین بسته‌شدن: {price(last['close'])}",
        f"تغییر آخرین کندل: {direction} ({change:+.5f})",
        analyze_structure(rows)
    ]

    highs, lows = find_swings(rows)

    if highs:
        output.append(
            f"آخرین Swing High: {price(highs[-1][1])}"
        )

    if lows:
        output.append(
            f"آخرین Swing Low: {price(lows[-1][1])}"
        )

    zones = find_fvg(rows)

    if zones:
        zone = zones[-1]
        age = len(rows) - 1 - zone["index"]

        output.append(
            f"آخرین FVG {zone['type']}: "
            f"{price(zone['low'])} - {price(zone['high'])}; "
            f"سن تقریبی: {age} کندل"
        )
    else:
        output.append("FVG اخیر: پیدا نشد")

    divergence = detect_hidden_divergence(rows)

    if divergence:
        output.append(
            f"هیدن دایورجنس: {divergence['direction']}\n"
            f"{divergence['description']}"
        )
    else:
        output.append(
            "هیدن دایورجنس چهار‌فازی: سیگنال معتبر پیدا نشد"
        )

    return "\n".join(output)


def symbol_name(symbol):
    return symbol.replace("=X", "")


# ============================================================
# MAIN
# ============================================================

def main():
    now = datetime.now(timezone.utc).astimezone(
        ZoneInfo("Asia/Tehran")
    )

    report = [
        "★★ Hadi Market Reports ★★",
        f"گزارش چندتایم‌فریمی: {now:%Y-%m-%d %H:%M} تهران",
        "MACD: 12,26,9",
        "تایم‌فریم‌ها: D1 | H4 | H1 | M15",
        "تحلیل: Market Structure | Swing | FVG | Hidden Divergence",
        "منبع: Yahoo Finance؛ قیمت ممکن است با بروکر تفاوت داشته باشد.",
        "H4 از داده H1 ساخته می‌شود.",
        ""
    ]

    cache = {}

    for symbol in SYMBOLS:
        report.append("==============================")
        report.append(f"● {symbol_name(symbol)}")

        try:
            for tf, interval, period in TIMEFRAMES:
                key = (symbol, interval)

                try:
                    if key not in cache:
                        cache[key] = fetch_data(
                            symbol, interval, period
                        )

                    rows = cache[key]
                    report.append(
                        analyze_timeframe(tf, rows)
                    )

                except Exception as exc:
                    report.append(
                        f"【{tf}】 خطا در دریافت داده: "
                        f"{str(exc)[:120]}"
                    )

            try:
                hourly_key = (symbol, "1h")

                if hourly_key not in cache:
                    cache[hourly_key] = fetch_data(
                        symbol, "1h", "60d"
                    )

                h4_rows = build_h4(cache[hourly_key])

                report.append(
                    analyze_timeframe("H4", h4_rows)
                )

            except Exception as exc:
                report.append(
                    f"【H4】 خطا در ساخت داده: {str(exc)[:120]}"
                )

        except Exception as exc:
            report.append(
                f"خطای نماد {symbol_name(symbol)}: {str(exc)[:120]}"
            )

        report.append("")

    message = "\n".join(report)

    send_telegram(message)


if __name__ == "__main__":
    main()