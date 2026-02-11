from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class QuotaResponse(BaseModel):
  leftPersonalQuota: str

@dataclass
class _LockedQuota(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def locked_left_quota(self, productId: str) -> str:
    """https://developers.binance.com/docs/simple_earn/account/Get-Locked-Personal-Left-Quota"""
    query = self.signed_query({
      'productId': productId,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.get(
      f'/sapi/v1/simple-earn/locked/personalLeftQuota?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, QuotaResponse).leftPersonalQuota
  