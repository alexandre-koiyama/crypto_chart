from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import ClientMixin
from binance.types import ErrorRoot, BinanceException, validate_response

class AvgPrice(BaseModel):
  mins: int
  """Avg. price interval"""
  price: str
  closeTime: int
  """Last trade time (millis timestamp)"""

@dataclass
class _AvgPrice(ClientMixin):
  @ClientMixin.with_client
  async def avg_price(
    self, pair: str,
  ) -> AvgPrice:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#current-average-price"""
    params  = {'symbol': pair }
    r = await self.client.get(f'/api/v3/avgPrice', params=params)
    return validate_response(r.text, AvgPrice)
  