import pytest
from binance.spot.data import MarketData

@pytest.mark.asyncio
async def test_order_book():
  client = MarketData()
  await client.order_book('BTCUSDT', limit=5)

@pytest.mark.asyncio
async def test_recent_trades():
  client = MarketData()
  await client.recent_trades('BTCUSDT', limit=5)

@pytest.mark.asyncio
async def test_old_trades():
  client = MarketData()
  await client.old_trades('BTCUSDT', limit=5)

@pytest.mark.asyncio
async def test_agg_trades():
  from datetime import datetime, timedelta
  client = MarketData()
  await client.agg_trades('BTCUSDT', limit=5)
  await client.agg_trades('BTCUSDT', limit=5, startTime=datetime.now() - timedelta(days=1))
  await client.agg_trades('BTCUSDT', limit=5, endTime=datetime.now() - timedelta(days=1))

@pytest.mark.asyncio
async def test_candles():
  from datetime import datetime, timedelta
  client = MarketData()
  await client.candles('BTCUSDT', interval='1m', limit=5)
  await client.candles('BTCUSDT', start=datetime.now() - timedelta(days=1), interval='1m', limit=5)

@pytest.mark.asyncio
async def test_ui_candles():
  from datetime import datetime, timedelta
  client = MarketData()
  await client.ui_candles('BTCUSDT', interval='1m', limit=5)
  await client.ui_candles('BTCUSDT', start=datetime.now() - timedelta(days=1), interval='1m', limit=5)

@pytest.mark.asyncio
async def test_avg_price():
  client = MarketData()
  await client.avg_price('BTCUSDT')

@pytest.mark.asyncio
async def test_stats_24h():
  client = MarketData()
  r = await client.price_stats_24h('SOLUSDC')
  assert 'SOLUSDC' in r
  r = await client.price_stats_24h('BTCUSDT', 'ETHUSDT')
  assert 'BTCUSDT' in r and 'ETHUSDT' in r

@pytest.mark.asyncio
async def test_stats_day():
  client = MarketData()
  r = await client.price_stats_day('SOLUSDC')
  assert 'SOLUSDC' in r
  r = await client.price_stats_day('BTCUSDT', 'ETHUSDT')
  assert 'BTCUSDT' in r and 'ETHUSDT' in r

@pytest.mark.asyncio
async def test_stats():
  client = MarketData()
  r = await client.price_stats('SOLUSDC')
  assert 'SOLUSDC' in r
  r = await client.price_stats('BTCUSDT', 'ETHUSDT', window='5m')
  assert 'BTCUSDT' in r and 'ETHUSDT' in r

@pytest.mark.asyncio
async def test_price():
  client = MarketData()
  r = await client.price('SOLUSDC')
  assert 'SOLUSDC' in r
  r = await client.price('BTCUSDT', 'ETHUSDT')
  assert 'BTCUSDT' in r and 'ETHUSDT' in r
