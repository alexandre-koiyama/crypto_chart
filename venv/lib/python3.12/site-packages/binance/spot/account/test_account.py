import pytest
from binance.spot.account import Account

@pytest.mark.asyncio
async def test_info():
  client = Account.env()
  await client.account_info()

@pytest.mark.asyncio
async def test_trades():
  from datetime import datetime, timedelta
  client = Account.env()
  await client.account_trades('BTCUSDT')
  await client.account_trades('BTCUSDT', start=datetime.now() - timedelta(days=1))
  await client.account_trades('BTCUSDT', end=datetime.now() - timedelta(days=1))