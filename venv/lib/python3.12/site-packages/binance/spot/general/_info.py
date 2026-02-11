from typing_extensions import TypeVar, Literal, Generic, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from binance.types import OrderType, ErrorRoot, BinanceException, validate_response
from binance.util import ClientMixin, encode_query

S = TypeVar('S', bound=str)

class PriceFilter(BaseModel):
  filterType: Literal['PRICE_FILTER']
  minPrice: str
  maxPrice: str
  tickSize: str

class LotSize(BaseModel):
  filterType: Literal['LOT_SIZE']
  minQty: str
  maxQty: str
  stepSize: str

class NotionalFilter(BaseModel):
  filterType: Literal['NOTIONAL']
  minNotional: str
  maxNotional: str
  applyToMarket: bool
  avgPriceMins: int

class IcebergFilter(BaseModel):
  filterType: Literal['ICEBERG_PARTS']
  limit: int

class OtherFilter(BaseModel):
  model_config = ConfigDict(extra='allow')
  filterType: str

Filter = PriceFilter | LotSize | NotionalFilter | IcebergFilter | OtherFilter 

class SymbolInfo(BaseModel):
  symbol: str
  status: Literal['TRADING', 'BREAK', 'HALT', 'AUCTION_MATCH', 'AUCTION_OPEN', 'AUCTION_CLOSE']
  baseAsset: str
  baseAssetPrecision: int
  quoteAsset: str
  quotePrecision: int
  quoteAssetPrecision: int
  baseCommissionPrecision: int
  quoteCommissionPrecision: int
  orderTypes: list[OrderType]
  icebergAllowed: bool
  ocoAllowed: bool
  quoteOrderQtyMarketAllowed: bool
  isSpotTradingAllowed: bool
  isMarginTradingAllowed: bool
  filters: list[Filter]

  @property
  def price_filter(self) -> PriceFilter:
    for f in self.filters:
      if f.filterType == 'PRICE_FILTER':
        return f # type: ignore
    raise RuntimeError('Price filter not found')
      
  @property
  def lot_size(self) -> LotSize:
    for f in self.filters:
      if f.filterType == 'LOT_SIZE':
        return f # type: ignore
    raise RuntimeError('Lot size filter not found')
  
  @property
  def notional(self) -> NotionalFilter:
    for f in self.filters:
      if f.filterType == 'NOTIONAL':
        return f # type: ignore
    raise RuntimeError('Notional filter not found')
  
  @property
  def iceberg(self) -> IcebergFilter:
    for f in self.filters:
      if f.filterType == 'ICEBERG_PARTS':
        return f # type: ignore
    raise RuntimeError('Iceberg filter not found')

class ExchangeInfoResponse(BaseModel):
  timezone: str
  serverTime: int
  """Millis timestamp"""
  symbols: list[SymbolInfo]

@dataclass
class ExchangeInfo(Generic[S]):
  timezone: str
  serverTime: int
  """Millis timestamp"""
  symbols: Mapping[S, SymbolInfo]

@dataclass
class _ExchangeInfo(ClientMixin):
  @ClientMixin.with_client
  async def exchange_info(self, symbol: S, *symbols: S) -> ExchangeInfo[S]:
    """https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-endpoints#exchange-information"""
    symbols = (symbol, *symbols)
    params = {'symbols': encode_query(symbols)}
    r = await self.client.get(f'/api/v3/exchangeInfo', params=params)
    info = validate_response(r.text, ExchangeInfoResponse)
    return ExchangeInfo(
      timezone=info.timezone,
      serverTime=info.serverTime,
      symbols={s.symbol: s for s in info.symbols if s.symbol in symbols}
    )