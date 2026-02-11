from typing_extensions import TypeVar, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, RootModel
from binance.util import ClientMixin, encode_query
from binance.types import validate_response

S = TypeVar('S', bound=str)

class PriceStats24h(BaseModel):
  symbol: str
  priceChange: str
  priceChangePercent: str
  weightedAvgPrice: str
  prevClosePrice: str
  lastPrice: str
  lastQty: str
  bidPrice: str
  bidQty: str
  askPrice: str
  askQty: str
  openPrice: str
  highPrice: str
  lowPrice: str
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

class PriceStats24hResponse(RootModel):
  root: list[PriceStats24h]

@dataclass
class _PriceStats24h(ClientMixin):
  @ClientMixin.with_client
  async def price_stats_24h(
    self, symbol: S, *symbols: S,
  ) -> Mapping[S, PriceStats24h]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#24hr-ticker-price-change-statistics"""
    params  = {'symbols': encode_query([symbol, *symbols]) }
    r = await self.client.get(f'/api/v3/ticker/24hr', params=params)
    stats = validate_response(r.text, PriceStats24hResponse).root
    ret = {}
    for s in stats:
      ret[s.symbol] = s
    return ret

  