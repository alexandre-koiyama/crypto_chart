from dataclasses import dataclass
from binance.util import UserMixin, timestamp
from binance.types import validate_response

@dataclass
class _CancelOrderList(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def cancel_order_list(self, symbol: str, orderListId: int):
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#cancel-order-list-trade"""
    query = self.signed_query({
      'symbol': symbol,
      'orderListId': orderListId,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.delete(
      f'/api/v3/orderList?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text)
  