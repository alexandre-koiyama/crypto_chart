from dataclasses import dataclass
from ._info import _AccountInfo
from ._trades import _AccountTrades

@dataclass
class Account(_AccountInfo, _AccountTrades):
  ...