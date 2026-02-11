from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class LockedProductDetail(BaseModel):
  asset: str
  rewardAsset: str | None = None
  duration: int
  renewable: bool
  isSoldOut: bool
  apr: str | None = None
  status: str
  subscriptionStartTime: int
  """Millis timestamp"""
  extraRewardAsset: str | None = None
  extraRewardAPR: str | None = None

class LockedProductQuota(BaseModel):
  totalPersonalQuota: str
  minimum: str

class LockedProduct(BaseModel):
  projectId: str
  detail: LockedProductDetail
  quota: LockedProductQuota

class LockedProductsResponse(BaseModel):
  rows: list[LockedProduct]

@dataclass
class _LockedProducts(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def locked_products(
    self, asset: str | None = None, *,
    page: int | None = None, size: int | None = None,
  ) -> list[LockedProduct]:
    """https://developers.binance.com/docs/simple_earn/account/Get-Simple-Earn-Locked-Product-List"""
    params: dict = {
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    }
    if asset is not None:
      params['asset'] = asset
    if page is not None:
      params['current'] = page
    if size is not None:
      params['size'] = size
    query = self.signed_query(params)
    r = await self.client.get(
      f'/sapi/v1/simple-earn/locked/list?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, LockedProductsResponse).rows
  