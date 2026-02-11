from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

@dataclass
class Balance:
  asset: str
  free: str
  locked: str

@dataclass
class CommissionRates:
  maker: str
  taker: str
  buyer: str
  seller: str

class AccountInfo(BaseModel):
  makerCommission: int
  takerCommission: int
  buyerCommission: int
  sellerCommission: int
  commissionRates: CommissionRates
  canTrade: bool
  canWithdraw: bool
  canDeposit: bool
  brokered: bool
  requireSelfTradePrevention: bool
  preventSor: bool
  updateTime: int
  """Millis timestamp"""
  accountType: str
  balances: list[Balance]
  uid: int

  def free(self, asset: str):
    from decimal import Decimal
    for b in self.balances:
      if b.asset == asset:
        return Decimal(b.free)
    return Decimal(0)

@dataclass
class _AccountInfo(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def account_info(self) -> AccountInfo:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/account-endpoints#account-information-user_data"""
    query = self.signed_query({
      'omitZeroBalances': 'true',
      'timestamp': timestamp.now(),
      'recvWindow': self.recvWindow,
    })
    r = await self.client.get(
      f'/api/v3/account?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, AccountInfo)