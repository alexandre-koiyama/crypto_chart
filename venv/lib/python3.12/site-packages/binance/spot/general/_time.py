from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import ClientMixin
from binance.types import ErrorRoot, BinanceException, validate_response

class ServerTime(BaseModel):
  serverTime: int
  """Millis timestamp"""

@dataclass
class _ServerTime(ClientMixin):
  @ClientMixin.with_client
  async def server_time(self) -> int:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-endpoints#check-server-time"""
    r = await self.client.get(f'/api/v3/time')
    return validate_response(r.text, ServerTime).serverTime

  