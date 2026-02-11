from dataclasses import dataclass
from datetime import datetime
from binance.util import ClientMixin, timestamp
from binance.types import validate_response
from ._candles import Candle, CandlesResponse

@dataclass
class _UiCandles(ClientMixin):
  @ClientMixin.with_client
  async def ui_candles(
    self, pair: str, start: datetime | None = None, *,
    interval: str, limit: int = 1000,
  ) -> list[Candle]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#uiklines"""
    params  = {'symbol': pair, 'interval': interval, 'limit': limit}
    if start is not None:
      params['startTime'] = timestamp.dump(start)
    r = await self.client.get(f'/api/v3/uiKlines', params=params)
    trades = validate_response(r.text, CandlesResponse).root
    return list(map(Candle.of, trades))
  