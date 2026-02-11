from dataclasses import dataclass
from binance.util import UserMixin, timestamp
from binance.types import Order, validate_response

@dataclass
class _TestOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def test_order(self, pair: str, order: Order):
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#test-new-order-trade"""
    query = self.signed_query({
      'symbol': pair,
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
      **order,
    })
    r = await self.client.post(
      f'/api/v3/order/test?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text)
  