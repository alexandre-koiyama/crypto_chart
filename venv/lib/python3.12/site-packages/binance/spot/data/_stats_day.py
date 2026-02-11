from typing_extensions import TypeVar, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, RootModel
from binance.util import ClientMixin, encode_query
from binance.types import validate_response

S = TypeVar('S', bound=str)

class PriceStats(BaseModel):
  symbol: str
  priceChange: str
  priceChangePercent: str
  weightedAvgPrice: str
  openPrice: str
  highPrice: str
  lowPrice: str
  lastPrice: str
  volume: str
  quoteVolume: str
  openTime: int
  """Millis timestamp"""
  closeTime: int
  """Millis timestamp"""
  firstId: int
  """First trade ID"""
  lastId: int
  """Last trade ID"""
  count: int
  """Trade count"""

class PriceStatsResponse(RootModel):
  root: list[PriceStats]

@dataclass
class _PriceStatsDay(ClientMixin):
  @ClientMixin.with_client
  async def price_stats_day(
    self, symbol: S, *symbols: S,
  ) -> Mapping[S, PriceStats]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#trading-day-ticker"""
    params  = {'symbols': encode_query([symbol, *symbols]) }
    r = await self.client.get(f'/api/v3/ticker/tradingDay', params=params)
    stats = validate_response(r.text, PriceStatsResponse).root
    ret = {}
    for s in stats:
      ret[s.symbol] = s
    return ret

  