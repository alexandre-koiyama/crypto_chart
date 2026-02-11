from typing_extensions import TypedDict, NotRequired, Literal
from binance.types import Side, TimeInForce


class BaseOrder(TypedDict):
  side: Side

class BaseLimitOrder(BaseOrder):
  quantity: str
  price: str
  timeInForce: TimeInForce

class LimitOrder(BaseLimitOrder):
  type: Literal['LIMIT']
  icebergQty: NotRequired[str]

class LimitMakerOrder(BaseLimitOrder):
  type: Literal['LIMIT_MAKER']
  icebergQty: NotRequired[str]

class MarketOrder(BaseOrder):
  type: Literal['MARKET']
  quantity: str
  timeInForce: NotRequired[TimeInForce]

class MarketOrderQuote(BaseOrder):
  type: Literal['MARKET']
  quoteOrderQty: str
  timeInForce: NotRequired[TimeInForce]

class BaseSL(BaseOrder):
  type: Literal['STOP_LOSS']
  quantity: str

class SLStop(BaseSL):
  stopPrice: str

class SLDelta(BaseSL):
  trailingDelta: str

StopLossOrder = SLStop | SLDelta

class SLLimitStop(BaseLimitOrder):
  type: Literal['STOP_LOSS_LIMIT']
  stopPrice: str

class SLLimitDelta(BaseLimitOrder):
  type: Literal['STOP_LOSS_LIMIT']
  trailingDelta: str

StopLossLimitOrder = SLLimitStop | SLLimitDelta

class BaseTP(BaseOrder):
  type: Literal['TAKE_PROFIT']
  quantity: str

class TPStop(BaseTP):
  stopPrice: str

class TPDelta(BaseTP):
  trailingDelta: str

TakeProfitOrder = TPStop | TPDelta

class TPLimitStop(BaseLimitOrder):
  type: Literal['TAKE_PROFIT_LIMIT']
  stopPrice: str

class TPLimitDelta(BaseLimitOrder):
  type: Literal['TAKE_PROFIT_LIMIT']
  trailingDelta: str

TakeProfitLimitOrder = TPLimitStop | TPLimitDelta

Order = LimitOrder | LimitMakerOrder | MarketOrder | MarketOrderQuote \
  | StopLossOrder | StopLossLimitOrder | TakeProfitOrder | TakeProfitLimitOrder
