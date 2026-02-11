from typing_extensions import TypeVar, Mapping
from dataclasses import dataclass
from binance.util import ClientMixin, encode_query
from binance.types import validate_response
from ._stats_day import PriceStats, PriceStatsResponse

S = TypeVar('S', bound=str)
@dataclass
class _PriceStats(ClientMixin):
  @ClientMixin.with_client
  async def price_stats(
    self, symbol: S, *symbols: S,
    window: str = '1d',
  ) -> Mapping[S, PriceStats]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#rolling-window-price-change-statistics"""
    params  = {'symbols': encode_query([symbol, *symbols]), 'windowSize': window }
    r = await self.client.get(f'/api/v3/ticker', params=params)
    stats = validate_response(r.text, PriceStatsResponse).root
    ret = {}
    for s in stats:
      ret[s.symbol] = s
    return ret

  