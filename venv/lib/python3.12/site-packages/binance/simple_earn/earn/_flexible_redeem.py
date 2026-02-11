from typing_extensions import Literal
from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class FlexibleRedeemResponse(BaseModel):
  redeemId: int
  success: bool

@dataclass
class _FlexibleRedeem(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def flexible_redeem(
    self, *, productId: str, amount: str | None = None,
    dest_account: Literal['SPOT', 'FUND'] | None = None,
  ) -> FlexibleRedeemResponse:
    """If `amount is None`, redeems all. https://developers.binance.com/docs/simple_earn/earn/Redeem-Flexible-Product"""
    params = {
      'productId': productId,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if dest_account is not None:
      params['destAccount'] = dest_account
    if amount is not None:
      params['amount'] = amount
    else:
      params['redeemAll'] = True
    query = self.signed_query(params)
    r = await self.client.post(
      f'/sapi/v1/simple-earn/flexible/redeem?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, FlexibleRedeemResponse)