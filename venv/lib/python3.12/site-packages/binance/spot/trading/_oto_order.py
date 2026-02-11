from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import Order, LimitOrder, LimitMakerOrder, MarketOrder, validate_response

class OrderResponse(BaseModel):
  orderId: int

class OtoResponse(BaseModel):
  orderListId: int
  transactTime: int | None = None
  """Millis timestamp"""
  orders: tuple[OrderResponse, OrderResponse]

  @property
  def workingId(self):
    return self.orders[0].orderId
  
  @property
  def pendingId(self):
    return self.orders[1].orderId


@dataclass
class _OtoOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def oto_order(
    self, pair: str, *,
    working: LimitOrder | LimitMakerOrder,
    pending: LimitOrder | MarketOrder | LimitMakerOrder
  ) -> OtoResponse:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-list---oto-trade"""

    def cap_first(s: str):
      return s[0].upper() + s[1:]

    def rename(order: Order, prefix: str) -> dict:
      """Turns keys e.g. `price` to `{prefix}Price`"""
      return {prefix + cap_first(key): value for key, value in order.items()}
    
    query = self.signed_query({
      'symbol': pair,
      'timestamp': timestamp.now(),
      'newOrderRespType': 'FULL',
      **rename(working, 'working'),
      **rename(pending, 'pending'),
    })
    r = await self.client.post(
      f'/api/v3/orderList/oto?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, OtoResponse)
  