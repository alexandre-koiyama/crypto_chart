"""
Simple Earn API [[docs](https://developers.binance.com/docs/simple_earn/)]
- Account [[docs](https://developers.binance.com/docs/simple_earn/account)]
- Earn [[docs](https://developers.binance.com/docs/simple_earn/earn)]
"""
from .earn import Earn
from .simple_earn import SimpleEarn

__all__ = ['Earn', 'SimpleEarn']