import sqlite3
import threading
from datetime import datetime
from config.settings import DB_PATH

class ThreadSafeDatabaseLogger:
    """Provides a thread-safe connection to record real-time metrics and order states"""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_schema()

    def _initialize_schema(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Order Execution Ledger
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_ledger (
                    id TEXT PRIMARY KEY, timestamp TEXT, symbol TEXT, 
                    side TEXT, qty INTEGER, price REAL, status TEXT
                )
            """)
            # Live Metrics Logging
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_telemetry (
                    timestamp TEXT, symbol TEXT, close_price REAL, 
                    net_worth REAL, reward REAL, action INTEGER
                )
            """)
            conn.commit()

    def log_order(self, order_id, symbol, side, qty, price, status="FILLED"):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO order_ledger VALUES (?, ?, ?, ?, ?, ?, ?)",
                (order_id, datetime.utcnow().isoformat(), symbol, side, qty, price, status)
            )

    def log_telemetry(self, symbol, close_price, net_worth, reward, action):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO agent_telemetry VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), symbol, close_price, net_worth, reward, action)
            )
