from typing_extensions import Any
from dataclasses import dataclass
from pydantic import RootModel
from datetime import datetime
from binance.util import ClientMixin, timestamp
from binance.types import validate_response



CandleArray = tuple[int, str, str, str, str, str, int, str, int, str, str, Any]

class CandleModel(RootModel):
  """[
    1499040000000,      // Kline open time
    "0.01634790",       // Open price
    "0.80000000",       // High price
    "0.01575800",       // Low price
    "0.01577100",       // Close price
    "148976.11427815",  // Volume
    1499644799999,      // Kline Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "0"                 // Unused field, ignore.
  ]"""
  root: CandleArray

class CandlesResponse(RootModel):
  root: list[CandleArray]

@dataclass
class Candle:
  open_time: datetime
  close_time: datetime
  open: str
  close: str
  high: str
  low: str
  base_volume: str
  quote_volume: str
  trades: int
  taker_buy_base_volume: str
  taker_buy_quote_volume: str

  @classmethod
  def of(cls, arr: CandleArray):
    return cls(
      open_time = timestamp.parse(arr[0]),
      close_time = timestamp.parse(arr[6]),
      open = arr[1],
      close = arr[4],
      high = arr[2],
      low = arr[3],
      base_volume = arr[5],
      quote_volume = arr[7],
      trades = int(arr[8]),
      taker_buy_base_volume = arr[9],
      taker_buy_quote_volume = arr[10],
    )

@dataclass
class _Candles(ClientMixin):
  @ClientMixin.with_client
  async def candles(
    self, pair: str, start: datetime | None = None, *,
    interval: str, limit: int = 1000,
  ) -> list[Candle]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data"""
    params  = {'symbol': pair, 'interval': interval, 'limit': limit}
    if start is not None:
      params['startTime'] = timestamp.dump(start)
    r = await self.client.get(f'/api/v3/klines', params=params)
    trades = validate_response(r.text, CandlesResponse).root
    return list(map(Candle.of, trades))
  