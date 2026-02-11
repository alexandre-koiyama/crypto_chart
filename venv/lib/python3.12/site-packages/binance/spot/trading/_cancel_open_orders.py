from dataclasses import dataclass
from binance.util import UserMixin, timestamp
from binance.types import validate_response

@dataclass
class _CancelOpenOrders(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def cancel_open_orders(self, symbol: str):
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#cancel-all-open-orders-on-a-symbol-trade"""
    query = self.signed_query({
      'symbol': symbol,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })

    r = await self.client.delete(
      f'/api/v3/openOrders?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    validate_response(r.text)
  