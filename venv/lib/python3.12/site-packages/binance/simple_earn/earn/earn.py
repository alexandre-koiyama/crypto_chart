from dataclasses import dataclass
from ._flexible_subscribe import _FlexibleSubscribe
from ._flexible_redeem import _FlexibleRedeem
from ._flexible_preview import _FlexiblePreview

@dataclass
class Earn(_FlexibleSubscribe, _FlexibleRedeem, _FlexiblePreview):
  ...