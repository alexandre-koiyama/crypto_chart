from typing_extensions import Literal
from dataclasses import dataclass
from pydantic import BaseModel, RootModel
from binance.util import UserMixin, timestamp
from binance.types import Order, validate_response
from ._new_order import NewOrderResponse

class ReplaceSuccess(BaseModel):
  cancelResult: Literal['SUCCESS']
  newOrderResult: Literal['SUCCESS']
  cancelResponse: NewOrderResponse
  newOrderResponse: NewOrderResponse

class ReplaceFailure(BaseModel):
  cancelResult: Literal['FAILURE']
  newOrderResult: Literal['NOT_ATTEMPTED']

class ReplacePartial(BaseModel):
  cancelResult: Literal['SUCCESS']
  newOrderResult: Literal['FAILURE']
  cancelResponse: NewOrderResponse

ReplaceResult = ReplaceSuccess | ReplaceFailure | ReplacePartial

class ReplaceFailureResponse(BaseModel):
  code: Literal[-2022]
  data: ReplaceFailure

class ReplacePartialResponse(BaseModel):
  code: Literal[-2021]
  data: ReplacePartial

class ReplaceError(RootModel):
  root: ReplaceFailureResponse | ReplacePartialResponse

@dataclass
class _ReplaceOrder(UserMixin):
  recvWindow: int = 5000

  @UserMixin.with_client
  async def replace_order(
    self, pair: str, orderId: int, order: Order, *,
    mode: Literal['STOP_ON_FAILURE', 'ALLOW_FAILURE'] = 'STOP_ON_FAILURE',
  ) -> ReplaceResult:
    query = self.signed_query({
      'symbol': pair,
      'cancelOrderId': orderId,
      'newOrderRespType': 'FULL',
      'timestamp': timestamp.now(),
      'cancelReplaceMode': mode,
      **order,
    })
    r = await self.client.post(
      f'/api/v3/order/cancelReplace?{query}',
      headers={'X-MBX-APIKEY': self.api_key},
    )
    obj = r.json()
    if obj.get('code') in [-2022, -2021]:
      return ReplaceError.model_validate(obj).root.data
    else:
      return validate_response(r.text, ReplaceSuccess)
  