from dataclasses import dataclass
from binance.util import ClientMixin
from binance.types import validate_response
from ._trades_recent import Trade, TradeResponse

@dataclass
class _OldTrades(ClientMixin):
  @ClientMixin.with_client
  async def old_trades(self, symbol: str, *, limit: int = 500, fromId: int | None = None) -> list[Trade]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#old-trade-lookup"""
    params = { 'symbol': symbol, 'limit': limit }
    if fromId is not None:
      params['fromId'] = fromId
    r = await self.client.get(f'/api/v3/historicalTrades', params=params)
    return validate_response(r.text, TradeResponse).root