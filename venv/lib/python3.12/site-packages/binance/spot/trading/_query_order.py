from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import OrderStatus, TimeInForce, Side, OrderType, validate_response

class QueryOrderResponse(BaseModel):
  orderId: int
  orderListId: int
  price: str
  origQty: str
  executedQty: str
  cummulativeQuoteQty: str
  status: OrderStatus
  timeInForce: TimeInForce
  type: OrderType
  side: Side
  stopPrice: str
  icebergQty: str
  time: int
  """Millis timestamp"""
  updateTime: int
  """Millis timestamp"""
  isWorking: bool
  origQuoteOrderQty: str


@dataclass
class _QueryOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def query_order(self, symbol: str, orderId: int) -> QueryOrderResponse:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#query-order-user_data"""
    query = self.signed_query({
      'symbol': symbol,
      'orderId': orderId,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.get(
      f'/api/v3/order?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, QueryOrderResponse)
  