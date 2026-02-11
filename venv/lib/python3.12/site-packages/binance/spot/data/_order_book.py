from dataclasses import dataclass
from pydantic import BaseModel
from binance.util import ClientMixin
from binance.types import validate_response

@dataclass
class Order:
  price: str
  qty: str

@dataclass
class OrderBook:
  lastUpdateId: int
  bids: list[Order]
  asks: list[Order]

class OrderBookResponse(BaseModel):
  lastUpdateId: int
  bids: list[tuple[str, str]]
  asks: list[tuple[str, str]]

@dataclass
class _OrderBook(ClientMixin):
  @ClientMixin.with_client
  async def order_book(self, symbol: str, *, limit: int = 100) -> OrderBook:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#order-book"""
    r = await self.client.get(f'/api/v3/depth', params={'symbol': symbol, 'limit': limit})
    data = validate_response(r.text, OrderBookResponse)
    return OrderBook(
      lastUpdateId=data.lastUpdateId,
      bids=[Order(price=p, qty=q) for p, q in data.bids],
      asks=[Order(price=p, qty=q) for p, q in data.asks]
    )