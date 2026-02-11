from dataclasses import dataclass
from ._new_order import _NewOrder
from ._test_order import _TestOrder
from ._query_order import _QueryOrder
from ._cancel_order import _CancelOrder
from ._cancel_open_orders import _CancelOpenOrders
from ._replace_order import _ReplaceOrder
from ._query_open_orders import _QueryOpenOrders
from ._query_all_orders import _QueryAllOrders

from ._oco_order import _OcoOrder
from ._oto_order import _OtoOrder
from ._otoco_order import _OtocoOrder
from ._cancel_order_list import _CancelOrderList


@dataclass
class Trading(
  _NewOrder, _TestOrder, _QueryOrder, _CancelOrder, _CancelOpenOrders,
  _ReplaceOrder, _QueryOpenOrders, _QueryAllOrders,
  _OcoOrder, _OtoOrder, _OtocoOrder, _CancelOrderList,
):
  ...