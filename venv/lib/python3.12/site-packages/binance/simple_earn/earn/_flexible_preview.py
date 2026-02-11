from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class FlexiblePreviewResponse(BaseModel):
  totalAmount: str
  rewardAsset: str
  airDropAsset: str | None = None
  estDailyBonusRewards: str
  estDailyRealTimeRewards: str
  estDailyAirdropRewards: str

@dataclass
class _FlexiblePreview(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def flexible_preview(self, *, productId: str, amount: str) -> FlexiblePreviewResponse:
    """https://developers.binance.com/docs/simple_earn/earn/Get-Flexible-Subscription-Preview"""
    params = {
      'productId': productId,
      'amount': amount,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    query = self.signed_query(params)
    r = await self.client.get(
      f'/sapi/v1/simple-earn/flexible/subscriptionPreview?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, FlexiblePreviewResponse)
  