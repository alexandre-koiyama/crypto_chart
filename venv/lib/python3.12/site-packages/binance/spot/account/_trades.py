from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, RootModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class Trade(BaseModel):
  symbol: str
  id: int
  orderId: int
  orderListId: int
  price: str
  qty: str
  quoteQty: str
  commission: str
  commissionAsset: str
  time: int
  """Millis timestamp"""
  isBuyer: bool
  isMaker: bool
  isBestMatch: bool

class TradesResponse(RootModel):
  root: list[Trade]

@dataclass
class _AccountTrades(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def account_trades(
    self, symbol: str, *, fromId: int | None = None, limit: int | None = None,
    start: datetime | None = None, end: datetime | None = None,
  ) -> list[Trade]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/account-endpoints#account-trade-list-user_data"""
    params = {
      'symbol': symbol,
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
    }
    if fromId is not None:
      params['fromId'] = fromId
    if limit is not None:
      params['limit'] = limit
    if start is not None:
      params['startTime'] = timestamp.dump(start)
    if end is not None:
      params['endTime'] = timestamp.dump(end)
    query = self.signed_query(params)
    r = await self.client.get(
      f'/api/v3/myTrades?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, TradesResponse).root