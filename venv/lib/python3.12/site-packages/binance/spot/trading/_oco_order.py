from typing_extensions import overload
from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response, Order, LimitMakerOrder, \
  StopLossOrder, StopLossLimitOrder, TakeProfitOrder, TakeProfitLimitOrder

class OrderResponse(BaseModel):
  orderId: int

class OcoResponse(BaseModel):
  orderListId: int
  transactTime: int
  """Millis timestamp"""
  orders: tuple[OrderResponse, OrderResponse]

  @property
  def aboveId(self):
    return self.orders[0].orderId
  
  @property
  def belowId(self):
    return self.orders[1].orderId


@dataclass
class _OcoOrder(UserMixin):
  recvWindow: int = 5000

  @overload
  async def oco_order(
    self, pair: str, *,
    above: LimitMakerOrder | TakeProfitOrder | TakeProfitLimitOrder,
    below: StopLossOrder | StopLossLimitOrder,
  ) -> OcoResponse:
    ...
  @overload
  async def oco_order(
    self, pair: str, *,
    above: StopLossOrder | StopLossLimitOrder,
    below: LimitMakerOrder | TakeProfitOrder | TakeProfitLimitOrder,
  ) -> OcoResponse:
    ...
  @UserMixin.with_client
  async def oco_order(self, pair: str, *, above: Order, below: Order) -> OcoResponse:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-list---oco-trade"""

    def cap_first(s: str):
      return s[0].upper() + s[1:]

    def rename(order: Order, prefix: str) -> dict:
      """Turns keys e.g. `price` to `{prefix}Price`"""
      return {prefix + cap_first(key): value for key, value in order.items()}
    
    query = self.signed_query({
      'symbol': pair,
      'timestamp': timestamp.now(),
      'newOrderRespType': 'FULL',
      **rename(above, 'above'),
      **rename(below, 'below'),
    })
    r = await self.client.post(
      f'/api/v3/orderList/oco?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, OcoResponse)
  