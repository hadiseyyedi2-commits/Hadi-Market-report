import os
import time
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# =========================================================
# Hadi Market Reports - Hidden Divergence Scanner
# Timeframes: D1, H4, H1, M15
# MACD: 12, 26, 9
# =========================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TEHRAN = ZoneInfo("Asia/Tehran")

SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "USDCAD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "EURCHF=X",
    "EURAUD=X",
    "EURNZD=X",
    "EURCAD=X",
    "GBPJPY=X",
    "GBPCHF=X",
    "GBPAUD=X",
    "GBPCAD=X",
    "GBPNZD=X",
    "AUDJPY=X",
    "AUDNZD=X",
    "AUDCAD=X",
    "NZDJPY=X",
    "XAUUSD=X",
    "XAGUSD=X",
]

# تایم‌فریم‌های مستقیم از Yahoo Finance
TIMEFRAMES = [
    ("D1", "1d", "1y"),
    ("H1", "1h", "60d"),
    ("M15", "15m", "60d"),
]

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0"
})


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN یا TELEGRAM_CHAT_ID تنظیم نشده است."
        )

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    chunks = []
    current = ""

    for line in message.splitlines():
        if len(current) + len(line) + 1 > 3900:
            if current:
                chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line

    if current:
        chunks.append(current)

    for chunk in chunks:
        response = SESSION.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": chunk,
            },
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):
            raise RuntimeError(
                f"Telegram API error: {result}"
            )

        time.sleep(0.3)


# =========================================================
# MARKET DATA
# =========================================================

def fetch_data(symbol, interval, period):
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + symbol
    )

    response = SESSION.get(
        url,
        params={
            "interval": interval,
            "range": period,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    result = (
        payload.get("chart", {})
        .get("result")
    )

    if not result:
        error = (
            payload.get("chart", {})
            .get("error")
        )
        raise ValueError(
            f"No market data: {error}"
        )

    item = result[0]

    timestamps = item.get("timestamp") or []

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
                "time": int(timestamp),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            })

        except (IndexError, TypeError, ValueError):
            continue

    rows.sort(key=lambda row: row["time"])

    return rows


# =========================================================
# BUILD H4 FROM H1
# =========================================================

def build_h4(hourly_rows):
    """
    ساخت کندل‌های چهار‌ساعته از داده‌های H1.
    فقط بلوک‌هایی نگه داشته می‌شوند که حداقل چهار ساعت
    داده را پوشش دهند.
    """

    buckets = {}

    for row in hourly_rows:
        bucket_time = (
            row["time"] // 14400
        ) * 14400

        if bucket_time not in buckets:
            buckets[bucket_time] = {
                "time": bucket_time,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "_first": row["time"],
                "_last": row["time"],
                "_count": 1,
            }

        else:
            candle = buckets[bucket_time]

            candle["high"] = max(
                candle["high"],
                row["high"],
            )

            candle["low"] = min(
                candle["low"],
                row["low"],
            )

            if row["time"] < candle["_first"]:
                candle["open"] = row["open"]
                candle["_first"] = row["time"]

            if row["time"] > candle["_last"]:
                candle["close"] = row["close"]
                candle["_last"] = row["time"]

            candle["_count"] += 1

    result = []

    for candle in sorted(
        buckets.values(),
        key=lambda x: x["time"],
    ):
        if (
            candle["_count"] >= 4
            and candle["_last"] - candle["time"] >= 3 * 3600
        ):
            result.append({
                "time": candle["time"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
            })

    return result


# =========================================================
# REMOVE CURRENT INCOMPLETE CANDLE
# =========================================================

def remove_incomplete_candle(rows, timeframe):
    """
    از استفاده از آخرین کندل ناقص جلوگیری می‌کند.
    زمان‌بندی Yahoo ممکن است با زمان‌بندی بروکر متفاوت باشد.
    """

    if not rows:
        return rows

    now = int(datetime.now(timezone.utc).timestamp())

    seconds = {
        "M15": 15 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
        "D1": 24 * 60 * 60,
    }[timeframe]

    # کندل فقط پس از پایان بازه زمانی آن قابل استفاده است.
    return [
        row for row in rows
        if row["time"] + seconds <= now
    ]


# =========================================================
# EMA AND MACD 12, 26, 9
# =========================================================

def ema(values, period):
    result = [None] * len(values)

    alpha = 2.0 / (period + 1.0)

    state = None

    for i, value in enumerate(values):
        if value is None:
            continue

        if state is None:
            state = value
        else:
            state = (
                alpha * value
                + (1.0 - alpha) * state
            )

        result[i] = state

    return result


def calculate_macd(closes):
    fast = ema(closes, 12)
    slow = ema(closes, 26)

    main = []

    for i in range(len(closes)):
        if fast[i] is None or slow[i] is None:
            main.append(None)
        else:
            main.append(
                fast[i] - slow[i]
            )

    signal = ema(main, 9)

    histogram = []

    for i in range(len(main)):
        if main[i] is None or signal[i] is None:
            histogram.append(None)
        else:
            histogram.append(
                main[i] - signal[i]
            )

    return main, signal, histogram


# =========================================================
# FOUR MACD ZERO-LINE PHASES
# =========================================================

def get_macd_phases(macd):
    """
    هر فاز با تغییر علامت خط اصلی MACD تعریف می‌شود.
    مقدار مثبت = بالای صفر
    مقدار منفی = پایین صفر
    """

    valid = [
        (i, value)
        for i, value in enumerate(macd)
        if value is not None
    ]

    if not valid:
        return []

    phases = []

    first_i, first_value = valid[0]

    sign = 1 if first_value >= 0 else -1

    points = [
        (first_i, first_value)
    ]

    for i, value in valid[1:]:
        new_sign = 1 if value >= 0 else -1

        if new_sign != sign:
            phases.append({
                "sign": sign,
                "points": points,
            })

            sign = new_sign

            points = [
                (i, value)
            ]

        else:
            points.append(
                (i, value)
            )

    phases.append({
        "sign": sign,
        "points": points,
    })

    return phases


# =========================================================
# HIDDEN DIVERGENCE DETECTION
# =========================================================

def detect_hidden_divergence(rows):
    """
    BUY:
        P1 -> N1 -> P2 -> N2

        Price Low(N2) > Price Low(N1)
        MACD Low(N2) < MACD Low(N1)

    SELL:
        N1 -> P1 -> N2 -> P2

        Price High(P2) < Price High(P1)
        MACD High(P2) > MACD High(P1)
    """

    if len(rows) < 60:
        return []

    closes = [
        row["close"]
        for row in rows
    ]

    macd, signal, histogram = calculate_macd(
        closes
    )

    phases = get_macd_phases(macd)

    if len(phases) < 4:
        return []

    signals = []

    # بررسی توالی‌های چهارفازی در تاریخچه
    for j in range(3, len(phases)):
        phase1, phase2, phase3, phase4 = (
            phases[j - 3:j + 1]
        )

        signs = (
            phase1["sign"],
            phase2["sign"],
            phase3["sign"],
            phase4["sign"],
        )

        # ---------------------------------------------
        # BULLISH HIDDEN DIVERGENCE
        # ---------------------------------------------

        if signs == (1, -1, 1, -1):

            # N1: پایین‌ترین کف قیمت در فاز منفی اول
            n1 = min(
                phase2["points"],
                key=lambda p: rows[p[0]]["low"],
            )

            # N2: پایین‌ترین کف قیمت در فاز منفی دوم
            n2 = min(
                phase4["points"],
                key=lambda p: rows[p[0]]["low"],
            )

            i1, macd1 = n1
            i2, macd2 = n2

            price1 = rows[i1]["low"]
            price2 = rows[i2]["low"]

            if (
                price2 > price1
                and macd2 < macd1
            ):
                signals.append({
                    "direction": "BUY",
                    "type": "Hidden Bullish Divergence",
                    "point1": {
                        "label": "N1",
                        "index": i1,
                        "time": rows[i1]["time"],
                        "price": price1,
                        "macd": macd1,
                    },
                    "point2": {
                        "label": "N2",
                        "index": i2,
                        "time": rows[i2]["time"],
                        "price": price2,
                        "macd": macd2,
                    },
                    "pattern_end": i2,
                })

        # ---------------------------------------------
        # BEARISH HIDDEN DIVERGENCE
        # ---------------------------------------------

        elif signs == (-1, 1, -1, 1):

            # P1: بالاترین سقف قیمت در فاز مثبت اول
            p1 = max(
                phase2["points"],
                key=lambda p: rows[p[0]]["high"],
            )

            # P2: بالاترین سقف قیمت در فاز مثبت دوم
            p2 = max(
                phase4["points"],
                key=lambda p: rows[p[0]]["high"],
            )

            i1, macd1 = p1
            i2, macd2 = p2

            price1 = rows[i1]["high"]
            price2 = rows[i2]["high"]

            if (
                price2 < price1
                and macd2 > macd1
            ):
                signals.append({
                    "direction": "SELL",
                    "type": "Hidden Bearish Divergence",
                    "point1": {
                        "label": "P1",
                        "index": i1,
                        "time": rows[i1]["time"],
                        "price": price1,
                        "macd": macd1,
                    },
                    "point2": {
                        "label": "P2",
                        "index": i2,
                        "time": rows[i2]["time"],
                        "price": price2,
                        "macd": macd2,
                    },
                    "pattern_end": i2,
                })

    # جدیدترین الگو در اولویت است.
    signals.sort(
        key=lambda item: item["pattern_end"],
        reverse=True,
    )

    # فقط جدیدترین سیگنال را برای هر جهت در این تایم‌فریم نگه می‌داریم.
    latest_by_direction = {}

    for signal_item in signals:
        direction = signal_item["direction"]

        if direction not in latest_by_direction:
            latest_by_direction[direction] = signal_item

    return list(
        latest_by_direction.values()
    )


# =========================================================
# FORMATTING
# =========================================================

def display_symbol(symbol):
    return symbol.replace("=X", "")


def format_price(value):
    if abs(value) >= 1000:
        return f"{value:.3f}"

    if abs(value) >= 100:
        return f"{value:.3f}"

    if abs(value) >= 1:
        return f"{value:.5f}"

    return f"{value:.6f}"


def format_time(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        timezone.utc,
    ).astimezone(TEHRAN).strftime(
        "%Y-%m-%d %H:%M"
    )


def format_signal(symbol, timeframe, signal):
    p1 = signal["point1"]
    p2 = signal["point2"]

    icon = (
        "🟢 BUY"
        if signal["direction"] == "BUY"
        else "🔻 SELL"
    )

    return "\n".join([
        "★★ HIDDEN DIVERGENCE ★★",
        f"نماد: {display_symbol(symbol)}",
        f"تایم‌فریم: {timeframe}",
        f"جهت: {icon}",
        f"نوع الگو: {signal['type']}",
        "",
        f"نقطه اول ({p1['label']}):",
        f"زمان کندل: {format_time(p1['time'])} تهران",
        f"قیمت نقطه: {format_price(p1['price'])}",
        f"MACD Main: {p1['macd']:.8f}",
        "",
        f"نقطه دوم ({p2['label']}):",
        f"زمان کندل: {format_time(p2['time'])} تهران",
        f"قیمت نقطه: {format_price(p2['price'])}",
        f"MACD Main: {p2['macd']:.8f}",
        "",
        "روش بررسی: چهار فاز متوالی MACD",
        "MACD: 12, 26, 9",
    ])


# =========================================================
# MAIN SCANNER
# =========================================================

def main():
    now = datetime.now(TEHRAN)

    print(
        f"Starting Hidden Divergence scan: {now:%Y-%m-%d %H:%M}"
    )

    cache = {}
    found_signals = []
    errors = []

    for symbol in SYMBOLS:
        print(f"Scanning {symbol}")

        for timeframe, interval, period in TIMEFRAMES:
            try:
                key = (symbol, interval)

                if key not in cache:
                    cache[key] = fetch_data(
                        symbol,
                        interval,
                        period,
                    )

                rows = cache[key]

                rows = remove_incomplete_candle(
                    rows,
                    timeframe,
                )

                if len(rows) < 60:
                    continue

                signals = detect_hidden_divergence(
                    rows
                )

                for signal in signals:
                    found_signals.append(
                        format_signal(
                            symbol,
                            timeframe,
                            signal,
                        )
                    )

            except Exception as exc:
                error_text = (
                    f"{display_symbol(symbol)} "
                    f"{timeframe}: {str(exc)[:120]}"
                )

                errors.append(error_text)

                print(error_text)

        # H4 از داده H1 ساخته می‌شود.
        try:
            hourly_key = (symbol, "1h")

            if hourly_key not in cache:
                cache[hourly_key] = fetch_data(
                    symbol,
                    "1h",
                    "60d",
                )

            h4_rows = build_h4(
                cache[hourly_key]
            )

            h4_rows = remove_incomplete_candle(
                h4_rows,
                "H4",
            )

            if len(h4_rows) >= 60:
                signals = detect_hidden_divergence(
                    h4_rows
                )

                for signal in signals:
                    found_signals.append(
                        format_signal(
                            symbol,
                            "H4",
                            signal,
                        )
                    )

        except Exception as exc:
            error_text = (
                f"{display_symbol(symbol)} "
                f"H4: {str(exc)[:120]}"
            )

            errors.append(error_text)

            print(error_text)

    if found_signals:
        header = [
            "★★ HADI HIDDEN DIVERGENCE REPORT ★★",
            f"زمان بررسی: {now:%Y-%m-%d %H:%M} تهران",
            "تایم‌فریم‌ها: D1 | H4 | H1 | M15",
            "MACD: 12,26,9",
            f"تعداد سیگنال‌ها: {len(found_signals)}",
            "",
        ]

        message = "\n\n".join(
            header + found_signals
        )

        send_telegram(message)

        print(
            f"Sent {len(found_signals)} signals to Telegram."
        )

    else:
        print(
            "No valid four-phase hidden divergence found. "
            "No Telegram message sent."
        )

    if errors:
        print(
            f"Completed with {len(errors)} data errors."
        )


if __name__ == "__main__":
    main()