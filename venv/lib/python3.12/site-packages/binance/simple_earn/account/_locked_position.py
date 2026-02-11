from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class LockedPosition(BaseModel):
  positionId: int
  parentPositionId: int
  projectId: str
  asset: str
  amount: str
  purchaseTime: int
  duration: str
  accrualDays: str
  rewardAsset: str
  APY: str
  rewardAmt: str
  extraRewardAsset: str
  extraRewardAPR: str
  estExtraRewardAmt: str
  nextPay: str
  nextPayDate: int
  payPeriod: str
  redeemAmountEarly: str
  rewardsEndDate: int
  deliverDate: int
  redeemPeriod: str
  redeemingAmt: str
  redeemTo: str
  partialAmtDeliverDate: int
  canRedeemEarly: bool
  canFastRedemption: bool
  autoSubscribe: bool
  type: str
  status: str
  canReStake: bool

class LockedPositionResponse(BaseModel):
  rows: list[LockedPosition]
@dataclass
class _LockedPosition(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def locked_position(
    self, asset: str | None = None, *, positionId: int | None = None,
    projectId: str | None = None, page: int | None = None, size: int | None = None,
  ) -> list[LockedPosition]:
    """https://developers.binance.com/docs/simple_earn/account/Get-Locked-Product-Position"""
    params: dict = {
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if asset is not None:
      params['asset'] = asset
    if positionId is not None:
      params['positionId'] = positionId
    if projectId is not None:
      params['projectId'] = projectId
    if page is not None:
      params['current'] = page
    if size is not None:
      params['size'] = size
    query = self.signed_query(params)
    r = await self.client.get(
      f'/sapi/v1/simple-earn/locked/position?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, LockedPositionResponse).rows
  