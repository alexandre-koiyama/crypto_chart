from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import UserMixin, timestamp
from binance.types import validate_response

class WithdrawResponse(BaseModel):
  id: str

@dataclass
class Wallet(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def withdraw(self, coin: str, /, *, network: str, address: str, amount: str) -> WithdrawResponse:
    query = self.signed_query({
      'coin': coin, 'network': network, 'address': address, 'amount': amount,
      'recvWindow': self.recvWindow,
      'timestamp': timestamp.now(),
    })
    r = await self.client.post(
      f'/sapi/v1/capital/withdraw/apply?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    return validate_response(r.text, WithdrawResponse)