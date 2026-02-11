from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class FlexibleProduct(BaseModel):
  asset: str
  latestAnnualPercentageRate: str
  tierAnnualPercentageRate: dict[str, str] | None = None
  airDropPercentageRate: str | None = None
  canPurchase: bool
  canRedeem: bool
  isSoldOut: bool
  hot: bool
  minPurchaseAmount: str
  productId: str
  subscriptionStartTime: int
  """Millis timestamp"""
  status: str

class FlexibleProductsResponse(BaseModel):
  rows: list[FlexibleProduct]

@dataclass
class _FlexibleProducts(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def flexible_products(
    self, asset: str | None = None, *,
    page: int | None = None, size: int | None = None,
  ) -> list[FlexibleProduct]:
    """https://developers.binance.com/docs/simple_earn/account/Get-Simple-Earn-Flexible-Product-List"""
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
      f'/sapi/v1/simple-earn/flexible/list?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, FlexibleProductsResponse).rows
  