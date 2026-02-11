from dataclasses import dataclass
from ._new_order import _NewOrder
from ._cancel_order import _CancelOrder
from ._oto_order import _OtoOrder
from ._cancel_order_list import _CancelOrderList
from ._query_order import _QueryOrder

@dataclass
class Trading(
  _NewOrder, _CancelOrder, _OtoOrder, _CancelOrderList,
  _QueryOrder,
):
  ...