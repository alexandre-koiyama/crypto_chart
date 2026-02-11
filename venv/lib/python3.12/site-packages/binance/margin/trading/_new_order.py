from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import Order, validate_response

class NewOrderResponse(BaseModel):
  orderId: int
  transactTime: int
  """Millis timestamp"""


@dataclass
class _NewOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def new_order(self, pair: str, order: Order, *, isolated: bool = False) -> NewOrderResponse:
    """https://developers.binance.com/docs/margin_trading/trade/Margin-Account-New-Order"""
    query = self.signed_query({
      'symbol': pair,
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
      'isIsolated': isolated,
      **order,
    })
    r = await self.client.post(
      f'/sapi/v1/margin/order?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, NewOrderResponse)
  