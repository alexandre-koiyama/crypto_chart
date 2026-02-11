from dataclasses import dataclass
from ._order_book import _OrderBook
from ._trades_recent import _RecentTrades
from ._trades_old import _OldTrades
from ._trades_agg import _AggTrades
from ._candles import _Candles
from ._candles_ui import _UiCandles
from ._avg_price import _AvgPrice
from ._stats_24h import _PriceStats24h
from ._stats_day import _PriceStatsDay
from ._price import _Price
from ._stats import _PriceStats

@dataclass
class MarketData(
  _OrderBook, _RecentTrades, _OldTrades, _AggTrades,
  _Candles, _UiCandles, _AvgPrice,
  _PriceStats24h, _PriceStatsDay,
  _Price, _PriceStats
):
  ...