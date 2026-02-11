import pytest
from binance.spot.general import General

@pytest.mark.asyncio
async def test_ping():
  client = General()
  await client.ping()

@pytest.mark.asyncio
async def test_time():
  client = General()
  await client.server_time()

@pytest.mark.asyncio
async def test_info():
  client = General()
  r = await client.exchange_info('SOLUSDC')
  assert 'SOLUSDC' in r.symbols
  r = await client.exchange_info('BTCUSDT', 'ETHUSDT')
  assert 'BTCUSDT' in r.symbols and 'ETHUSDT' in r.symbols