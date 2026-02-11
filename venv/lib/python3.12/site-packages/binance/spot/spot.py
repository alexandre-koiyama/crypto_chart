from dataclasses import dataclass
from .general import General
from .data import MarketData
from .trading import Trading
from .account import Account

@dataclass
class Spot(General, MarketData, Trading, Account):
  ...
