import time
import datetime as dt

import pandas as pd
from binance.client import Client
from binance.helpers import date_to_milliseconds, interval_to_milliseconds

client = Client()  # public endpoints used


def get_klines(symbol: str, interval: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    limit = 1500

    if start_date:
        start_ts = int(date_to_milliseconds(start_date))
    else:
        now_ms = int(dt.datetime.utcnow().timestamp() * 1000)
        start_ts = now_ms - interval_to_milliseconds(interval) * limit * 2

    end_ts = int(dt.datetime.utcnow().timestamp() * 1000) if not end_date else int(date_to_milliseconds(end_date))
    cur = start_ts

    rows: list[list] = []
    step = interval_to_milliseconds(interval)

    while cur < end_ts:
        batch = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=cur,
            endTime=end_ts,
            limit=limit,
        )
        if not batch:
            break

        rows.extend(batch)
        newest_open = batch[-1][0]
        cur = newest_open + step
        if newest_open >= end_ts:
            break
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignored",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(None)
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


_SYMBOLS_CACHE = {"ts": 0.0, "rows": []}
_SYMBOLS_TTL_SEC = 300


def get_usdt_perp_symbols_by_24h_quote_volume(*, ascending: bool = True, limit: int = 300) -> list[str]:
    now = time.time()

    rows: list[tuple[str, float]]
    if _SYMBOLS_CACHE["rows"] and (now - _SYMBOLS_CACHE["ts"] < _SYMBOLS_TTL_SEC):
        rows = _SYMBOLS_CACHE["rows"]
    else:
        ex = client.futures_exchange_info()
        tradable = {
            s["symbol"]
            for s in ex.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
        }

        tickers = client.futures_ticker()
        rows = []
        for t in tickers:
            sym = t.get("symbol")
            if sym in tradable:
                try:
                    qv = float(t.get("quoteVolume") or 0.0)
                except Exception:
                    qv = 0.0
                rows.append((sym, qv))

        _SYMBOLS_CACHE["ts"] = now
        _SYMBOLS_CACHE["rows"] = rows

    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=not ascending)
    syms = [s for s, _ in rows_sorted]
    return syms[:limit]