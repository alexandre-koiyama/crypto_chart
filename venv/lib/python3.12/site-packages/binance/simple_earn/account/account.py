from dataclasses import dataclass
from ._account import _Account
from ._flexible_products import _FlexibleProducts
from ._locked_products import _LockedProducts
from ._flexible_position import _FlexiblePosition
from ._locked_position import _LockedPosition
from ._flexible_quota import _FlexibleQuota
from ._locked_quota import _LockedQuota


@dataclass
class Account(
  _Account, _FlexibleProducts, _LockedProducts,
  _FlexiblePosition, _LockedPosition,
  _FlexibleQuota, _LockedQuota
):
  ...