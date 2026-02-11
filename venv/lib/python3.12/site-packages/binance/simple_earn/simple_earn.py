from dataclasses import dataclass
from .earn import Earn
from .account import Account

@dataclass
class SimpleEarn(Earn, Account):
  ...
