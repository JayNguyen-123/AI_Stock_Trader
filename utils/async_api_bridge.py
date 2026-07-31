import aiohttp
import json
from config.settings import BROKER_API_KEY, BROKER_SECRET_KEY, BROKER_BASE_URL

class AsyncBrokerTradingBridge:
    """Asynchronous broker execution gateway using non-blocking connection reuse loops"""
    def __init__(self, base_url=BROKER_BASE_URL):
        self.headers = {
            "APCA-API-KEY-ID": BROKER_API_KEY,
            "APCA-API-SECRET-KEY": BROKER_SECRET_KEY,
            "Content-Type": "application/json"
        }
        self.orders_url = f"{base_url}/v2/orders"
        self.positions_url = f"{base_url}/v2/positions"
        self.session = None

    async def initialize_session(self):
        """Creates the long-lived client session footprint"""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close_session(self):
        """Cleans up connection allocations on system shutdown routines"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_position_qty(self, symbol):
        """Asynchronously checks active inventory limits for a given asset"""
        await self.initialize_session()
        url = f"{self.positions_url}/{symbol.upper()}"
        try:
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return int(data.get('qty', 0))
                return 0
        except Exception as e:
            print(f"[ASYNC BRIDGE ERROR] Position check timeout for {symbol}: {e}")
            return 0

    async def dispatch_market_order(self, symbol, side, quantity=1):
        """Asynchronously dispatches a market order without blocking parallel script executions"""
        await self.initialize_session()
        payload = {
            "symbol": symbol.upper(),
            "qty": str(quantity),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day"
        }
        try:
            async with self.session.post(self.orders_url, json=payload, timeout=5) as response:
                if response.status in (200, 202): # Fixed: Added successful HTTP status codes
                    order_receipt = await response.json()
                    print(f"[ASYNC ORDER FILLED] {symbol.upper()} | Side: {side.upper()} | ID: {order_receipt.get('id')}")
                    return order_receipt
                else:
                    error_text = await response.text()
                    print(f"[ASYNC ORDER ERROR] Server rejected trade execution: {error_text}")
                    return None
        except Exception as e:
            print(f"[ASYNC CRITICAL FAIL] Execution connection failure: {e}")
            return None
