from typing_extensions import TypeVar, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, RootModel
from binance.util import ClientMixin, encode_query
from binance.types import validate_response

S = TypeVar('S', bound=str)

class SymbolPrice(BaseModel):
  symbol: str
  price: str

class SymbolResponse(RootModel):
  root: list[SymbolPrice]

@dataclass
class _Price(ClientMixin):
  @ClientMixin.with_client
  async def price(
    self, symbol: S, *symbols: S,
  ) -> Mapping[S, str]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#symbol-price-ticker"""
    params  = {'symbols': encode_query([symbol, *symbols]) }
    r = await self.client.get(f'/api/v3/ticker/price', params=params)
    prices = validate_response(r.text, SymbolResponse).root
    ret = {}
    for p in prices:
      ret[p.symbol] = p.price
    return ret

  