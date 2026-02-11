from dataclasses import dataclass
from pydantic import RootModel
from binance.util import ClientMixin
from binance.types import validate_response

@dataclass
class Trade:
  id: int
  price: str
  qty: str
  quoteQty: str
  time: int
  """Millis timestamp"""
  isBuyerMaker: bool
  isBestMatch: bool

class TradeResponse(RootModel):
  root: list[Trade]

@dataclass
class _RecentTrades(ClientMixin):
  @ClientMixin.with_client
  async def recent_trades(self, symbol: str, *, limit: int = 500) -> list[Trade]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#recent-trades-list"""
    r = await self.client.get(f'/api/v3/trades', params={'symbol': symbol, 'limit': limit})
    return validate_response(r.text, TradeResponse).root
  