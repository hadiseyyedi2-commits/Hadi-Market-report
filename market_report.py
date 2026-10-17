import os
import requests
from datetime import datetime, timezone

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD",
    "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD",
    "GBPNZD", "AUDJPY", "AUDNZD",
    "AUDCAD", "NZDJPY"
]

YAHOO_SYMBOLS = {pair: pair + "=X" for pair in FOREX_PAIRS}
YAHOO_SYMBOLS["GOLD"] = "GC=F"
YAHOO_SYMBOLS["SILVER"] = "SI=F"


def get_market_data(symbol):
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + symbol
    )
    response = requests.get(
        url,
        params={"range": "5d", "interval": "1h"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=25
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("chart", {}).get("result")

    if not results:
        raise ValueError("No market data returned")

    item = results[0]
    timestamps = item.get("timestamp", [])
    quote = item["indicators"]["quote"][0]

    candles = []

    for i, timestamp in enumerate(timestamps):
        if i >= len(quote["close"]):
            continue

        close = quote["close"][i]
        if close is None:
            continue

        candles.append({
            "time": datetime.fromtimestamp(
                timestamp, timezone.utc
            ),
            "open": quote["open"][i],
            "high": quote["high"][i],
            "low": quote["low"][i],
            "close": close
        })

    if len(candles) < 3:
        raise ValueError("Not enough candles")

    return candles


def analyze(name, candles):
    last = candles[-1]
    previous = candles[-2]

    change = (
        (last["close"] - previous["close"])
        / previous["close"] * 100
    )

    recent = candles[-6:]
    highest = max(c["high"] for c in recent if c["high"] is not None)
    lowest = min(c["low"] for c in recent if c["low"] is not None)

    if last["close"] > previous["close"]:
        direction = "صعودی"
    elif last["close"] < previous["close"]:
        direction = "نزولی"
    else:
        direction = "بدون تغییر"

    return (
        f"{name}\n"
        f"آخرین قیمت: {last['close']}\n"
        f"تغییر کندل: {change:.3f}%\n"
        f"جهت آخرین کندل: {direction}\n"
        f"بالاترین قیمت ۶ کندل: {highest}\n"
        f"پایین‌ترین قیمت ۶ کندل: {lowest}"
    )


def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    for start in range(0, len(message), 3500):
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message[start:start + 3500]
            },
            timeout=25
        )
        response.raise_for_status()

        result = response.json()
        if not result.get("ok"):
            raise RuntimeError(
                "Telegram error: " + str(result)
            )


def main():
    now = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    report = [
        "📊 گزارش آزمایشی بازار",
        f"زمان تهیه: {now}",
        "تایم‌فریم داده: یک‌ساعته",
        "این گزارش اولیه است و سیگنال ورود نیست.",
        ""
    ]

    failures = []

    for name, symbol in YAHOO_SYMBOLS.items():
        try:
            candles = get_market_data(symbol)
            report.append(analyze(name, candles))
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    if failures:
        report.extend(["", "خطا در دریافت داده:"])
        report.extend(failures)

    send_telegram("\n\n".join(report))


if __name__ == "__main__":
    main()