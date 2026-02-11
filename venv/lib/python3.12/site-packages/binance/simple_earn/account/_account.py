from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class AccountSummary(BaseModel):
  totalAmountInBTC: str
  totalAmountInUSDT: str
  totalFlexibleAmountInBTC: str
  totalFlexibleAmountInUSDT: str
  totalLockedInBTC: str
  totalLockedInUSDT: str

@dataclass
class _Account(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def account(self) -> AccountSummary:
    """https://developers.binance.com/docs/simple_earn/account"""
    query = self.signed_query({
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.get(
      f'/sapi/v1/simple-earn/account?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, AccountSummary)
  