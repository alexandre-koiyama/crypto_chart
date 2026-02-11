from dataclasses import dataclass
from binance.util import UserMixin, timestamp
from binance.types import validate_response

@dataclass
class _CancelOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def cancel_order(self, symbol: str, orderId: int, *, isolated: bool = False):
    """https://developers.binance.com/docs/margin_trading/trade/Margin-Account-Cancel-Order"""
    query = self.signed_query({
      'symbol': symbol,
      'orderId': orderId,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
      'isIsolated': isolated,
    })

    r = await self.client.delete(
      f'/sapi/v1/margin/order?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text)
  