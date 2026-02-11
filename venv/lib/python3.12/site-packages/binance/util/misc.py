from typing_extensions import AsyncIterable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_DOWN, ROUND_FLOOR
from haskellian import ManagedAsync

T = TypeVar('T')

class timestamp:
  @staticmethod
  def parse(time: int) -> datetime:
    """Parse a Binance-issued millis timestamp"""
    return datetime.fromtimestamp(time/1e3)
  
  @staticmethod
  def dump(dt: datetime) -> int:
    """Dump a datetime object to a Binance-ready millis timestamp"""
    return int(1e3*dt.timestamp())
  
  @staticmethod
  def now() -> int:
    """Get the current time in Binance-ready millis timestamp"""
    return timestamp.dump(datetime.now())

def round2tick(x: Decimal, tick_size: Decimal) -> Decimal:
  r = (x / tick_size).quantize(Decimal('1.'), rounding=ROUND_HALF_DOWN) * tick_size
  return r.normalize()

def trunc2tick(x: Decimal, tick_size: Decimal) -> Decimal:
  r = (x / tick_size).to_integral_value(rounding=ROUND_FLOOR) * tick_size
  return r.normalize()

@dataclass
class Stream(Generic[T]):
  stream: AsyncIterable[T]
  subscribers: list[ManagedAsync[T]] = field(default_factory=list)

  async def run(self):
    async for item in self.stream:
      for subscriber in self.subscribers:
        subscriber.push(item)

  def subscribe(self) -> ManagedAsync[T]:
    sub = ManagedAsync[T]()
    self.subscribers.append(sub)
    return sub