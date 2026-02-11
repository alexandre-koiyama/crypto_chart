from typing_extensions import Literal
from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class FlexibleSubscribeResponse(BaseModel):
  purchaseId: int
  success: bool

@dataclass
class _FlexibleSubscribe(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def flexible_subscribe(
    self, *, productId: str, amount: str, auto_subscribe: bool | None = None,
    source_account: Literal['SPOT', 'FUND', 'ALL'] | None = None,
  ) -> FlexibleSubscribeResponse:
    """https://developers.binance.com/docs/simple_earn/earn"""
    params = {
      'productId': productId,
      'amount': amount,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if source_account is not None:
      params['sourceAccount'] = source_account
    if auto_subscribe is not None:
      params['autoSubscribe'] = auto_subscribe
    query = self.signed_query(params)
    r = await self.client.post(
      f'/sapi/v1/simple-earn/flexible/subscribe?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, FlexibleSubscribeResponse)
  