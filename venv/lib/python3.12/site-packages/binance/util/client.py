from typing_extensions import TypeVar, ParamSpec
from dataclasses import dataclass, field
from functools import wraps
import httpx

T = TypeVar('T', covariant=True)
P = ParamSpec('P')

@dataclass
class ClientMixin:
  base: str = field(default='https://api.binance.com', kw_only=True)

  async def __aenter__(self):
    self._client = httpx.AsyncClient(base_url=self.base)
    return self
  
  async def __aexit__(self, *args):
    if self._client is not None:
      await self._client.aclose()
      self._client = None

  @property
  def client(self) -> httpx.AsyncClient:
    client = getattr(self, '_client', None)
    if client is None:
      raise RuntimeError('Please use as context manager: `async with ...: ...`')
    return client
  
  @staticmethod
  def with_client(fn):
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
      if getattr(self, '_client', None) is None:
        async with self:
          return await fn(self, *args, **kwargs)
      else:
        return await fn(self, *args, **kwargs)
      
    return wrapper
