from typing_extensions import Literal, overload
from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class NewOrderResponse(BaseModel):
  orderId: int
  transactTime: int
  """Millis timestamp"""


class BorrowResponse(BaseModel):
  tranId: int

@dataclass
class Leasing(UserMixin):
  recvWindow: int = 5000

  @overload
  async def lease(self, type: Literal['BORROW', 'REPAY'], *, asset: str, amount: str, symbol: str, isolated: Literal[True]) -> BorrowResponse:
    """Borrow or repay an isolated-margin asset"""
  @overload
  async def lease(self, type: Literal['BORROW', 'REPAY'], *, asset: str, amount: str, isolated: Literal[False] = False) -> BorrowResponse:
    """Borrow or repay a cross-margin asset"""
  @UserMixin.with_client
  async def lease(self, type: Literal['BORROW', 'REPAY'], *, asset: str, symbol: str | None = None, amount: str, isolated: bool = False) -> BorrowResponse:
    """Borrow or repay a cross- or isolated-margin asset"""
    params = {
      'asset': asset,
      'amount': amount,
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
      'type': type,
    }
    if symbol:
      params['symbol'] = symbol
    if isolated:
      params['isIsolated'] = 'TRUE'
    query = self.signed_query(params)
    r = await self.client.post(
      f'/sapi/v1/margin/borrow-repay?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, BorrowResponse)