import pandas as pd


def add_indicators(df: pd.DataFrame, *, bb_period: int = 20, bb_std: float = 2.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ma = df["close"].rolling(bb_period, min_periods=bb_period).mean()
    sd = df["close"].rolling(bb_period, min_periods=bb_period).std(ddof=0)

    df["bb_mid"] = ma
    df["bb_upper"] = ma + bb_std * sd
    df["bb_lower"] = ma - bb_std * sd
    return df
