from dataclasses import dataclass
from ._info import _ExchangeInfo
from ._ping import _Ping
from ._time import _ServerTime

@dataclass
class General(_ExchangeInfo, _Ping, _ServerTime):
  ...