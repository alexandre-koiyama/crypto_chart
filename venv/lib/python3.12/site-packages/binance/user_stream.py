from typing_extensions import AsyncIterable, Literal
from dataclasses import dataclass
import asyncio
from datetime import timedelta
import traceback
from pydantic import BaseModel, RootModel, Field, ValidationError
import httpx
from websockets.asyncio.client import connect
from binance.types import Side, OrderType, OrderStatus, TimeInForce

class Balance(BaseModel):
  a: str
  """Asset"""
  f: float
  """Free amount"""
  l: float
  """Locked amount"""

class AccountUpdate(BaseModel):
  e: Literal['outboundAccountPosition']
  E: int
  """Event time (millis timestamp)"""
  u: int
  """Update time (millis timestamp)"""
  B: list[Balance]
  """Balances array (changed assets only)"""
  
class BalanceUpdate(BaseModel):
  e: Literal['balanceUpdate']
  E: int
  """Event time (millis timestamp)"""
  a: str
  """Asset"""
  d: float
  """Balance change (delta)"""
  T: int
  """Clear time (millis timestamp)"""

ExecutionType = Literal['NEW', 'CANCELED', 'REPLACED', 'REJECTED', 'TRADE', 'EXPIRED', 'TRADE_PREVENTION']

class OrderUpdate(BaseModel):
  e: Literal['executionReport']
  E: int
  """Event time (millis timestamp)"""
  s: str
  """Symbol"""
  c: str
  """Client order ID"""
  S: Side
  o: OrderType
  f: TimeInForce
  q: float
  """Order quantity"""
  p: float
  """Order price"""
  F: float
  """Iceberg quantity"""
  x: ExecutionType
  """Current execution type"""
  X: OrderStatus
  """Current order status"""
  i: int
  """Order ID"""

class ListStatus(BaseModel):
  e: Literal['listStatus']
  E: int
  """Event time (millis timestamp)"""
  s: str
  """Symbol"""
  g: int
  """OrderListId"""

Update = AccountUpdate | BalanceUpdate | OrderUpdate | ListStatus

class UpdateRoot(RootModel):
  root: Update = Field(discriminator='e')

@dataclass
class UserStream:
  api_key: str
  base: str = 'https://api.binance.com'
  ws_base: str = 'wss://stream.binance.com:9443'

  async def _create_stream(self) -> str:
    async with httpx.AsyncClient(base_url=self.base) as client:
      r = await client.post('/api/v3/userDataStream', headers={'X-MBX-APIKEY': self.api_key})
      r.raise_for_status()
      return r.json()['listenKey']
    
  async def _delete_stream(self, listen_key: str):
    async with httpx.AsyncClient(base_url=self.base) as client:
      r = await client.delete('/api/v3/userDataStream', headers={'X-MBX-APIKEY': self.api_key}, params={'listenKey': listen_key})
      r.raise_for_status()

  async def _ping_stream(self, listen_key: str):
    async with httpx.AsyncClient(base_url=self.base) as client:
      r = await client.put('/api/v3/userDataStream', headers={'X-MBX-APIKEY': self.api_key}, params={'listenKey': listen_key})
      r.raise_for_status()

  async def _keep_alive(self, listen_key: str, *, interval: timedelta = timedelta(minutes=30)):
    while True:
      await self._ping_stream(listen_key)
      await asyncio.sleep(interval.total_seconds())

  async def subscribe(self) -> AsyncIterable[Update]:
    listen_key = await self._create_stream()
    asyncio.create_task(self._keep_alive(listen_key))

    try:
      async with connect(f'{self.ws_base}/ws/{listen_key}') as ws:
        async for msg in ws:
          try:
            yield UpdateRoot.model_validate_json(msg).root
          except ValidationError:
            print('Error parsing message:', msg)
            traceback.print_exc()
    finally:
      await self._delete_stream(listen_key)
