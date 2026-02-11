"""
Spot API [[docs](https://developers.binance.com/docs/binance-spot-api-docs)]
- General [[docs](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-endpoints)]
- Market Data [[docs](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)]
- Trading [[docs](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints)]
- Account [[docs](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/account-endpoints)]
"""
from .general import General
from .data import MarketData
from .trading import Trading
from .account import Account
from .spot import Spot

__all__ = ['General', 'MarketData', 'Trading', 'Account', 'Spot']