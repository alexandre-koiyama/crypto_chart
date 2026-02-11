from dataclasses import dataclass
from binance.util import UserMixin, timestamp
from binance.types import validate_response

@dataclass
class _CancelOrderList(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def cancel_order_list(self, symbol: str, orderListId: int, *, isolated: bool = False):
    """https://developers.binance.com/docs/margin_trading/trade/Margin-Account-Cancel-OCO"""
    query = self.signed_query({
      'symbol': symbol,
      'orderListId': orderListId,
      'recvWindow': self.recvWindow,
      'isIsolated': isolated,
      'timestamp': timestamp.now(),
    })
    r = await self.client.delete(
      f'/sapi/v1/margin/orderList?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text)
  