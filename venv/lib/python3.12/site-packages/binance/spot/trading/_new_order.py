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
  async def new_order(self, pair: str, order: Order) -> NewOrderResponse:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-trade"""
    query = self.signed_query({
      'symbol': pair,
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
      **order,
    })
    r = await self.client.post(
      f'/api/v3/order?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, NewOrderResponse)
  