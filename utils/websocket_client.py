import asyncio
import json
import websockets
from config.settings import WS_STREAM_URL, BROKER_API_KEY, BROKER_SECRET_KEY

class AsyncMarketStreamClient:
    """Manages raw multi-asset market data pipelines over a single WebSocket connection"""
    def __init__(self, symbols, data_callback):
        self.symbols = symbols
        self.data_callback = data_callback # Dispatches valid ticks to the live tracker
        self.running = True

    async def connect_and_listen(self):
        async for websocket in websockets.connect(WS_STREAM_URL):
            try:
                # 1. Authenticate with the broker data streaming API
                auth_payload = {"action": "auth", "key": BROKER_API_KEY, "secret": BROKER_SECRET_KEY}
                await websocket.send(json.dumps(auth_payload))
                
                # 2. Subscribe to trades ('t') for all selected symbols
                sub_payload = {"action": "subscribe", "trades": self.symbols}
                await websocket.send(json.dumps(sub_payload))
                print(f"[WebSocket] Connected. Subscribed to market streams for: {self.symbols}")

                # 3. Stream data loop
                while self.running:
                    message = await websocket.recv()
                    data_packets = json.loads(message)
                    
                    for packet in data_packets:
                        # Map exchange variables ('t' = trade notification, 'T' = ticker symbol)
                        if packet.get('T') == 't':
                            symbol = packet['i']
                            price = float(packet['p'])
                            volume = int(packet['s'])
                            self.data_callback(symbol, price, volume)

            except websockets.ConnectionClosed:
                print("[WebSocket WARNING] Network dropped. Attempting automatic reconnection...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[WebSocket CRITICAL ERROR] Pipeline fault: {e}")
                await asyncio.sleep(1)

