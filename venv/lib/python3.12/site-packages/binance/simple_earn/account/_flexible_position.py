from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class FlexiblePosition(BaseModel):
  totalAmount: str
  tierAnnualPercentageRate: dict[str, float] | None = None
  latestAnnualPercentageRate: str
  yesterdayAirdropPercentageRate: str | None = None
  asset: str
  airDropAsset: str | None = None
  canRedeem: bool
  collateralAmount: str
  productId: str
  yesterdayRealTimeRewards: str
  cumulativeBonusRewards: str | None = None
  cumulativeRealTimeRewards: str
  cumulativeTotalRewards: str
  autoSubscribe: bool


class FlexiblePositionResponse(BaseModel):
  rows: list[FlexiblePosition]
@dataclass
class _FlexiblePosition(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def flexible_position(
    self, asset: str | None = None, *, productId: str | None = None,
    page: int | None = None, size: int | None = None,
  ) -> list[FlexiblePosition]:
    """https://developers.binance.com/docs/simple_earn/account/Get-Flexible-Product-Position"""
    params: dict = {
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if asset is not None:
      params['asset'] = asset
    if productId is not None:
      params['productId'] = productId
    if page is not None:
      params['current'] = page
    query = self.signed_query(params)
    r = await self.client.get(
      f'/sapi/v1/simple-earn/flexible/position?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, FlexiblePositionResponse).rows
  