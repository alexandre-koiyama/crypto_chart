from dataclasses import dataclass
from datetime import datetime
from pydantic import RootModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response
from ._query_order import QueryOrderResponse

class QueryAllOrdersResponse(RootModel):
  root: list[QueryOrderResponse]

@dataclass
class _QueryAllOrders(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def query_all_orders(
    self, symbol: str, *, limit: int | None = None,
    start: datetime | None = None, end: datetime | None = None,
  ) -> list[QueryOrderResponse]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#all-orders-user_data"""
    params = {
      'symbol': symbol,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if limit is not None:
      params['limit'] = limit
    if start is not None:
      params['startTime'] = timestamp.dump(start)
    if end is not None:
      params['endTime'] = timestamp.dump(end)
    query = self.signed_query(params)
    r = await self.client.get(
      f'/api/v3/openOrders?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, QueryAllOrdersResponse).root
  