import math
import os
from typing import Any

import pandas as pd


def maybe_place_market_order_from_df(
    df: pd.DataFrame,
    *,
    client: Any,
    symbol: str,
    usdt_to_use: float,
    leverage: int,
    take_profit_pct: float = 7.0,
    stop_loss_pct: float = 10.0,
) -> dict:
    """
    Places a USDT-M futures MARKET order if the last df['order'] is 'buy' or 'sell',
    and attaches TP/SL (% from entry).
    """
    if df is None or df.empty or "order" not in df.columns:
        return {"placed": False, "reason": "missing df/order"}

    if os.getenv("TRADING_ENABLED") != "1":
        return {"placed": False, "reason": "TRADING_DISABLED (set TRADING_ENABLED=1)"}

    last = str(df["order"].iloc[-1]).strip().lower()
    if last not in {"buy", "sell"}:
        return {"placed": False, "reason": f"no signal ({last})"}

    side = "BUY" if last == "buy" else "SELL"
    return place_futures_market_order(
        client,
        symbol=symbol,
        side=side,
        usdt_to_use=usdt_to_use,
        leverage=leverage,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )


def place_futures_market_order(
    client: Any,
    *,
    symbol: str,
    side: str,  # "BUY" or "SELL"
    usdt_to_use: float,
    leverage: int,
    take_profit_pct: float = 7.0,
    stop_loss_pct: float = 10.0,
) -> dict:
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        return {"placed": False, "reason": f"invalid side: {side}"}

    client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

    info = client.futures_exchange_info()
    sym_info = next(s for s in info["symbols"] if s["symbol"] == symbol)

    step_size = float(next(f for f in sym_info["filters"] if f["filterType"] == "LOT_SIZE")["stepSize"])
    tick_size = float(next(f for f in sym_info["filters"] if f["filterType"] == "PRICE_FILTER")["tickSize"])

    price_info = client.futures_mark_price(symbol=symbol)
    entry_price = float(price_info["markPrice"])

    notional = float(usdt_to_use) * int(leverage)
    raw_qty = notional / entry_price
    qty = math.floor(raw_qty / step_size) * step_size
    if qty <= 0:
        return {"placed": False, "reason": "qty<=0", "raw_qty": raw_qty, "step_size": step_size}

    entry_order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=qty,
    )

    exit_side = "SELL" if side == "BUY" else "BUY"

    def _round_down_to_tick(p: float) -> float:
        return math.floor(p / tick_size) * tick_size

    def _round_up_to_tick(p: float) -> float:
        return math.ceil(p / tick_size) * tick_size

    # TP/SL prices (% from entry)
    if side == "BUY":
        tp_price = entry_price * (1 + take_profit_pct / 100.0)
        sl_price = entry_price * (1 - stop_loss_pct / 100.0)
        tp_price = _round_down_to_tick(tp_price)
        sl_price = _round_up_to_tick(sl_price)
    else:  # SELL
        tp_price = entry_price * (1 - take_profit_pct / 100.0)
        sl_price = entry_price * (1 + stop_loss_pct / 100.0)
        tp_price = _round_up_to_tick(tp_price)
        sl_price = _round_down_to_tick(sl_price)

    tp_order = client.futures_create_order(
        symbol=symbol,
        side=exit_side,
        type="TAKE_PROFIT_MARKET",
        stopPrice=tp_price,
        closePosition=True,
        workingType="MARK_PRICE",
    )

    sl_order = client.futures_create_order(
        symbol=symbol,
        side=exit_side,
        type="STOP_MARKET",
        stopPrice=sl_price,
        closePosition=True,
        workingType="MARK_PRICE",
    )

    return {
        "placed": True,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "tp_pct": take_profit_pct,
        "sl_pct": stop_loss_pct,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "entry_order": entry_order,
        "tp_order": tp_order,
        "sl_order": sl_order,
    }