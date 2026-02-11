import asyncio
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.binance import get_klines, get_usdt_perp_symbols_by_24h_quote_volume
from app.indicators import add_indicators

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _norm(s: str) -> str | None:
    s = (s or "").strip()
    return s or None


def _to_json_floats(s: pd.Series) -> list[float | None]:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return [float(x) if np.isfinite(x) else None for x in arr]


async def _load_df(symbol: str, interval: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    df = await asyncio.to_thread(get_klines, symbol, interval, start_date, end_date)
    return add_indicators(df)


def _add_channels_and_signals(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    block: int = 50,
    max_blocks: int = 10,
    q: float = 0.90,  # outlier-robust (upper=q, lower=1-q)
    slope_eps: float = 1e-12,
) -> None:
    n = len(df)
    blocks = min(max_blocks, n // block)

    for i in range(blocks):
        end = n - i * block
        start = end - block
        d = df.iloc[start:end].copy().reset_index(drop=True)
        if d.empty:
            continue

        x = np.arange(len(d), dtype=float)
        close = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(d["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(d["low"], errors="coerce").to_numpy(dtype=float)

        msk = np.isfinite(close)
        if msk.sum() < 2:
            continue

        m, b = np.polyfit(x[msk], close[msk], 1)
        if abs(m) <= slope_eps:
            allow_sell = allow_buy = False
        else:
            allow_sell = m < 0
            allow_buy = m > 0

        base = m * x + b
        hi_res = (high - base)
        lo_res = (low - base)

        hi_res = hi_res[np.isfinite(hi_res)]
        lo_res = lo_res[np.isfinite(lo_res)]
        if len(hi_res) < 5 or len(lo_res) < 5:
            continue

        upper = base + float(np.nanquantile(hi_res, q))
        lower = base + float(np.nanquantile(lo_res, 1 - q))

        t = d["close_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        is_last = (i == 0)
        color = "rgba(0,0,0,0.85)" if is_last else "rgba(0,0,0,0.25)"
        dash = "dash" if is_last else "dot"

        fig.add_trace(go.Scatter(x=t, y=upper, mode="lines", name="Channel High", line=dict(width=1, dash=dash, color=color), showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=lower, mode="lines", name="Channel Low", line=dict(width=1, dash=dash, color=color), showlegend=False))

        sell_marks = np.full(len(d), np.nan, dtype=float)
        buy_marks = np.full(len(d), np.nan, dtype=float)

        if len(d) >= 2:
            h0, h1 = high[:-1], high[1:]
            u0, u1 = upper[:-1], upper[1:]
            l0, l1 = low[:-1], low[1:]
            d0, d1 = lower[:-1], lower[1:]

            sell = (np.isfinite(h0) & np.isfinite(h1) & np.isfinite(u0) & np.isfinite(u1) & (h0 <= u0) & (h1 > u1))
            buy = (np.isfinite(l0) & np.isfinite(l1) & np.isfinite(d0) & np.isfinite(d1) & (l0 >= d0) & (l1 < d1))

            if not allow_sell:
                sell[:] = False
            if not allow_buy:
                buy[:] = False

            sell_marks[1:][sell] = high[1:][sell]
            buy_marks[1:][buy] = low[1:][buy]

        fig.add_trace(go.Scatter(
            x=t,
            y=[float(v) if np.isfinite(v) else None for v in sell_marks],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", color="red", size=10),
            showlegend=is_last,
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=[float(v) if np.isfinite(v) else None for v in buy_marks],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
            showlegend=is_last,
        ))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    symbols = await asyncio.to_thread(get_usdt_perp_symbols_by_24h_quote_volume, ascending=True, limit=300)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "symbols": symbols,
            "symbol": "DOGEUSDT",
            "interval": "1m",
            "start_date": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end_date": dt.date.today().isoformat(),
            "chart_html": "",
            "title": "",
        },
    )


@router.post("/plot", response_class=HTMLResponse)
async def plot_post(
    request: Request,
    symbol: str = Form(...),
    interval: str = Form("1m"),
    start_date: str = Form(""),
    end_date: str = Form(""),
):
    symbol = symbol.strip().upper()
    start_date_n = _norm(start_date)
    end_date_n = _norm(end_date) or dt.date.today().isoformat()

    symbols = await asyncio.to_thread(get_usdt_perp_symbols_by_24h_quote_volume, ascending=True, limit=300)
    df = await _load_df(symbol, interval, start_date_n, end_date_n)

    if df is None or df.empty:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "symbols": symbols,
                "symbol": symbol,
                "interval": interval,
                "start_date": start_date_n or "",
                "end_date": end_date_n,
                "chart_html": "",
                "title": f"{symbol} {interval} (no data)",
            },
        )

    t = df["close_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["close"]), name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["high"]), name="High", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["low"]), name="Low", mode="lines", line=dict(width=1)))

    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["bb_upper"]), name="BB Upper", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["bb_mid"]), name="BB Mid", mode="lines", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=_to_json_floats(df["bb_lower"]), name="BB Lower", mode="lines", line=dict(width=1)))

    _add_channels_and_signals(fig, df, block=50, max_blocks=10, q=0.90)

    fig.update_layout(margin=dict(t=20, r=20, b=40, l=60), height=520)
    chart_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "symbols": symbols,
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date_n or "",
            "end_date": end_date_n,
            "chart_html": chart_html,
            "title": f"{symbol} {interval} {start_date_n or ''} → {end_date_n}",
        },
    )
