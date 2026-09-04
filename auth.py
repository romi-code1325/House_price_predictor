"""
auth.py
Lightweight username/password auth for the hackathon demo.
Uses a local SQLite database (users.db) and salted SHA-256 hashes.

NOTE: this is "good enough for a demo", not production-grade security.
No password reset, no email verification, no rate limiting.
"""

import sqlite3
import hashlib
import os
import secrets
import datetime

USERS_DB = "users.db"


def _get_conn():
    conn = sqlite3.connect(USERS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            sqft REAL,
            bedrooms INTEGER,
            bathrooms REAL,
            house_age REAL,
            distance_km REAL,
            school_rating INTEGER,
            crime_index REAL,
            predicted_price REAL,
            price_per_sqft REAL
        )
    """)
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return False, "Username already exists."

        salt = secrets.token_hex(8)
        pw_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
            (username, salt, pw_hash),
        )
        conn.commit()
        return True, "Account created. You can log in now."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if not row:
            return False
        salt, stored_hash = row
        return _hash_password(password, salt) == stored_hash
    finally:
        conn.close()


def save_prediction(username: str, features: dict, predicted_price: float) -> None:
    """
    Logs one prediction to history.
    `features` should have keys: sqft, bedrooms, bathrooms, house_age,
    distance_km, school_rating, crime_index.
    """
    price_per_sqft = predicted_price / features["sqft"] if features.get("sqft") else None
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO predictions (
                username, created_at, sqft, bedrooms, bathrooms, house_age,
                distance_km, school_rating, crime_index, predicted_price, price_per_sqft
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username.strip(),
            datetime.datetime.now().isoformat(timespec="seconds"),
            features.get("sqft"),
            features.get("bedrooms"),
            features.get("bathrooms"),
            features.get("house_age"),
            features.get("distance_km"),
            features.get("school_rating"),
            features.get("crime_index"),
            predicted_price,
            price_per_sqft,
        ))
        conn.commit()
    finally:
        conn.close()


def get_history(username: str, limit: int = 50):
    """Returns most recent predictions for a user, newest first, as a list of dicts."""
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT created_at, sqft, bedrooms, bathrooms, house_age,
                   distance_km, school_rating, crime_index,
                   predicted_price, price_per_sqft
            FROM predictions
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
        """, (username.strip(), limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_history(username: str) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM predictions WHERE username = ?", (username.strip(),))
        conn.commit()
    finally:
        conn.close()
