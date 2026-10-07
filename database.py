"""
database.py
Handles all MySQL operations: connection, insertion, and querying.
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, List, Dict, Any

import config


class DatabaseManager:
    """Manages the MySQL connection and CRUD operations for detection logs."""

    def __init__(self):
        self.connection: Optional[mysql.connector.MySQLConnection] = None
        self._connect()

    # ── Connection ──────────────────────────────────────────────────────────
    def _connect(self) -> None:
        """Establish a connection to the MySQL database."""
        try:
            self.connection = mysql.connector.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
                autocommit=True,
            )
            if self.connection.is_connected():
                print(f"[DB] Connected to `{config.DB_NAME}` on "
                      f"{config.DB_HOST}:{config.DB_PORT}")
        except Error as e:
            print(f"[DB ERROR] Could not connect to database: {e}")
            self.connection = None

    def is_connected(self) -> bool:
        """Return True if the connection is alive."""
        return self.connection is not None and self.connection.is_connected()

    def reconnect(self) -> None:
        """Re-establish the connection if it was dropped."""
        if not self.is_connected():
            self._connect()

    def close(self) -> None:
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("[DB] Connection closed.")

    # ── Insert ──────────────────────────────────────────────────────────────
    def insert_detection(
        self,
        object_class: str,
        confidence: float,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
    ) -> bool:
        """
        Insert a single detection event into the database.

        Returns:
            True if insert succeeded, False otherwise.
        """
        if not self.is_connected():
            self.reconnect()
            if not self.is_connected():
                return False

        query = f"""
            INSERT INTO {config.DB_TABLE}
                (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            datetime.now(),
            object_class,
            round(confidence, 4),
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
        )

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, values)
            cursor.close()
            return True
        except Error as e:
            print(f"[DB ERROR] Insert failed: {e}")
            return False

    # ── Batch Insert (for efficiency) ───────────────────────────────────────
    def insert_detections_batch(self, detections: List[Dict[str, Any]]) -> int:
        """
        Insert multiple detections at once.

        Args:
            detections: List of dicts with keys:
                        object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h

        Returns:
            Number of rows successfully inserted.
        """
        if not detections:
            return 0

        if not self.is_connected():
            self.reconnect()
            if not self.is_connected():
                return 0

        query = f"""
            INSERT INTO {config.DB_TABLE}
                (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
        """
        now = datetime.now()
        values_list = [
            (
                now,
                d["object_class"],
                round(d["confidence"], 4),
                d["bbox_x"],
                d["bbox_y"],
                d["bbox_w"],
                d["bbox_h"],
            )
            for d in detections
        ]

        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, values_list)
            inserted = cursor.rowcount
            cursor.close()
            return inserted
        except Error as e:
            print(f"[DB ERROR] Batch insert failed: {e}")
            return 0

    # ── Query ───────────────────────────────────────────────────────────────
    def fetch_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch the most recent `limit` detection logs."""
        if not self.is_connected():
            self.reconnect()
            if not self.is_connected():
                return []

        query = f"""
            SELECT log_id, timestamp, object_class, confidence,
                   bbox_x, bbox_y, bbox_w, bbox_h
            FROM {config.DB_TABLE}
            ORDER BY timestamp DESC
            LIMIT %s
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            print(f"[DB ERROR] Fetch failed: {e}")
            return []

    def fetch_class_counts(self) -> Dict[str, int]:
        """Return a dict of {object_class: count} for the current session."""
        if not self.is_connected():
            self.reconnect()
            if not self.is_connected():
                return {}

        query = f"""
            SELECT object_class, COUNT(*) AS cnt
            FROM {config.DB_TABLE}
            GROUP BY object_class
            ORDER BY cnt DESC
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            return {row["object_class"]: row["cnt"] for row in rows}
        except Error as e:
            print(f"[DB ERROR] Class count fetch failed: {e}")
            return {}

    def fetch_total_detections(self) -> int:
        """Return the total number of logged detections."""
        if not self.is_connected():
            self.reconnect()
            if not self.is_connected():
                return 0

        query = f"SELECT COUNT(*) AS total FROM {config.DB_TABLE}"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result["total"] if result else 0
        except Error as e:
            print(f"[DB ERROR] Total count fetch failed: {e}")
            return 0


# ── Convenience singleton ───────────────────────────────────────────────────
def get_db() -> DatabaseManager:
    """Return a new DatabaseManager instance."""
    return DatabaseManager()