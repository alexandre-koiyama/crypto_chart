from dataclasses import dataclass
from pydantic import RootModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response
from ._query_order import QueryOrderResponse

class QueryOpenOrdersResponse(RootModel):
  root: list[QueryOrderResponse]

@dataclass
class _QueryOpenOrders(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def query_open_orders(self, symbol: str) -> list[QueryOrderResponse]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#current-open-orders-user_data"""
    query = self.signed_query({
      'symbol': symbol,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.get(
      f'/api/v3/openOrders?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, QueryOpenOrdersResponse).root
  