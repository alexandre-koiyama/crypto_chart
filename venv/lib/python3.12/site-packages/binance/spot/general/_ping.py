from dataclasses import dataclass
from binance.util import ClientMixin
from binance.types import validate_response

@dataclass
class _Ping(ClientMixin):
  @ClientMixin.with_client
  async def ping(self):
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-endpoints#test-connectivity"""
    r = await self.client.get(f'/api/v3/ping')
    validate_response(r.text)
    

  