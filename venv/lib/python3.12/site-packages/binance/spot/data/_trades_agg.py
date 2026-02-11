from dataclasses import dataclass
from pydantic import BaseModel, RootModel
from datetime import datetime
from binance.util import ClientMixin, timestamp
from binance.types import validate_response

class AggTradeModel(BaseModel):
  a: int
  """Aggregate tradeId"""
  p: str
  """Price"""
  q: str
  """Quantity"""
  f: int
  """First tradeId"""
  l: int
  """Last tradeId"""
  T: int
  """Millis timestamp"""
  m: bool
  """Was the buyer the maker?"""
  M: bool
  """Was the trade the best price match?"""

@dataclass
class AggTrade:
  id: int
  price: str
  qty: str
  firstId: int
  lastId: int
  time: datetime
  isBuyerMaker: bool
  isBestMatch: bool

  @classmethod
  def of(cls, model: AggTradeModel):
    return cls(
      model.a,
      model.p,
      model.q,
      model.f,
      model.l,
      timestamp.parse(model.T),
      model.m,
      model.M
    )

class AggTradeResponse(RootModel):
  root: list[AggTradeModel]

@dataclass
class _AggTrades(ClientMixin):
  @ClientMixin.with_client
  async def agg_trades(
    self, symbol: str, *, limit: int = 500, fromId: int | None = None,
    startTime: datetime | None = None, endTime: datetime | None = None,
  ) -> list[AggTrade]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#compressedaggregate-trades-list"""
    params = { 'symbol': symbol, 'limit': limit }
    if fromId is not None:
      params['fromId'] = fromId
    if startTime is not None:
      params['startTime'] = timestamp.dump(startTime)
    if endTime is not None:
      params['endTime'] = timestamp.dump(endTime)
    r = await self.client.get(f'/api/v3/aggTrades', params=params)
    trades = validate_response(r.text, AggTradeResponse).root
    return list(map(AggTrade.of, trades))