"""
SQLite database tools for inventory management.
Handles stock checking, inventory subtraction, and restocking operations.
"""

import sqlite3
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

# Database file path
DB_PATH = Path(__file__).parent.parent / "inventory.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection. Auto-initializes database if it doesn't exist or is empty."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Check if inventory table exists, if not, initialize database
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inventory'")
        table_exists = cursor.fetchone() is not None
        if not table_exists:
            conn.close()
            init_database()
            # Reconnect after initialization
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
    except Exception:
        # If there's any error checking, close and reinitialize
        conn.close()
        init_database()
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
    
    return conn


def init_database() -> None:
    """Initialize SQLite database with inventory table and seed data."""
    # Create connection directly (avoid recursion)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create inventory table with price column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            item_name TEXT PRIMARY KEY,
            quantity INTEGER NOT NULL DEFAULT 0,
            price REAL NOT NULL DEFAULT 0.0
        )
    """)
    
    # Add price column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE inventory ADD COLUMN price REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Check if database is already seeded
    cursor.execute("SELECT COUNT(*) FROM inventory")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Seed database with sample items (item_name, quantity, price)
        sample_items = [
            ("tshirt", 10, 19.99),
            ("jeans", 5, 49.99),
            ("shoes", 8, 79.99),
            ("hat", 15, 14.99),
            ("jacket", 3, 99.99),
        ]
        cursor.executemany(
            "INSERT INTO inventory (item_name, quantity, price) VALUES (?, ?, ?)",
            sample_items
        )
        conn.commit()
        print(f"✅ Database initialized with {len(sample_items)} items")
    else:
        # Update existing items with default prices if they don't have prices
        default_prices = {
            "tshirt": 19.99,
            "jeans": 49.99,
            "shoes": 79.99,
            "hat": 14.99,
            "jacket": 99.99,
        }
        for item_name, price in default_prices.items():
            cursor.execute(
                "UPDATE inventory SET price = ? WHERE item_name = ? AND (price IS NULL OR price = 0)",
                (price, item_name)
            )
        conn.commit()
        print(f"✅ Database already initialized with {count} items")
    
    conn.close()


def check_stock(item_name: str) -> int:
    """
    Get available quantity for an item.
    Uses normalized item name for matching.
    
    Args:
        item_name: Name of the item to check
        
    Returns:
        Available quantity (0 if item doesn't exist)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    normalized_name = normalize_item_name(item_name)
    
    # Try exact match first
    cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (item_name,))
    result = cursor.fetchone()
    
    # If not found, try normalized name
    if not result and normalized_name != item_name:
        cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (normalized_name,))
        result = cursor.fetchone()
    
    # If still not found, try case-insensitive search
    if not result:
        cursor.execute("SELECT quantity FROM inventory WHERE LOWER(item_name) = LOWER(?)", (item_name,))
        result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else 0


def normalize_item_name(item_name: str) -> str:
    """
    Normalize item name for database lookup.
    Converts to lowercase, removes hyphens/spaces, handles plurals.
    
    Args:
        item_name: Original item name
        
    Returns:
        Normalized item name
    """
    if not item_name:
        return ""
    
    # Convert to lowercase
    normalized = item_name.lower().strip()
    
    # Remove hyphens and spaces
    normalized = normalized.replace("-", "").replace(" ", "")
    
    # Database stores these items as-is (some plural, some singular)
    db_item_names = ["tshirt", "jeans", "shoes", "hat", "jacket"]
    
    # Handle common variations that should map to database names
    name_mappings = {
        "tshirts": "tshirt",
        "t-shirts": "tshirt",
        "t_shirts": "tshirt",
        "tshirt": "tshirt",
        "shoe": "shoes",  # Database has "shoes" (plural)
        "shoes": "shoes",
        "hats": "hat",
        "hat": "hat",
        "jackets": "jacket",
        "jacket": "jacket",
        "jeans": "jeans",
        "jean": "jeans",  # Database has "jeans" (plural)
    }
    
    if normalized in name_mappings:
        normalized = name_mappings[normalized]
    elif normalized in db_item_names:
        # Already matches a database name
        pass
    else:
        # Try to match by removing trailing 's' if it exists
        if normalized.endswith("s"):
            singular = normalized[:-1]
            if singular in db_item_names:
                normalized = singular
            elif singular in name_mappings:
                normalized = name_mappings[singular]
    
    return normalized


def check_stock_multiple(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Batch check stock for multiple items.
    Uses normalized item names for matching.
    
    Args:
        items: List of dicts with 'item_name' key
        
    Returns:
        Dictionary mapping original item_name to available quantity
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    results = {}
    for item in items:
        original_name = item.get("item_name", "")
        normalized_name = normalize_item_name(original_name)
        
        # Try exact match first (original name)
        cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (original_name,))
        result = cursor.fetchone()
        
        # If not found, try normalized name
        if not result and normalized_name != original_name:
            cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (normalized_name,))
            result = cursor.fetchone()
        
        # If still not found, try case-insensitive search
        if not result:
            cursor.execute("SELECT quantity FROM inventory WHERE LOWER(item_name) = LOWER(?)", (original_name,))
            result = cursor.fetchone()
        
        results[original_name] = result[0] if result else 0
    
    conn.close()
    return results


def subtract_inventory(item_name: str, quantity: int) -> bool:
    """
    Subtract inventory from database.
    
    Args:
        item_name: Name of the item
        quantity: Quantity to subtract
        
    Returns:
        True if successful, False if insufficient stock or item doesn't exist
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check current stock
    cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (item_name,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return False
    
    current_qty = result[0]
    if current_qty < quantity:
        conn.close()
        return False
    
    # Subtract inventory
    new_qty = current_qty - quantity
    cursor.execute(
        "UPDATE inventory SET quantity = ? WHERE item_name = ?",
        (new_qty, item_name)
    )
    conn.commit()
    conn.close()
    
    return True


def restock_item(item_name: str, quantity: int) -> bool:
    """
    Restock an item in the database.
    
    Args:
        item_name: Name of the item to restock
        quantity: Quantity to add back
        
    Returns:
        True if successful, False if item doesn't exist
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if item exists
    cursor.execute("SELECT quantity FROM inventory WHERE item_name = ?", (item_name,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return False
    
    # Add inventory back
    current_qty = result[0]
    new_qty = current_qty + quantity
    cursor.execute(
        "UPDATE inventory SET quantity = ? WHERE item_name = ?",
        (new_qty, item_name)
    )
    conn.commit()
    conn.close()
    
    return True


def get_item_price(item_name: str) -> float:
    """
    Get the price of an item.
    
    Args:
        item_name: Name of the item
        
    Returns:
        Price of the item (0.0 if item doesn't exist)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT price FROM inventory WHERE item_name = ?", (item_name,))
    result = cursor.fetchone()
    conn.close()
    
    return float(result[0]) if result and result[0] is not None else 0.0


def get_items_prices(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Get prices for multiple items.
    
    Args:
        items: List of dicts with 'item_name' key
        
    Returns:
        Dictionary mapping item_name to price
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    results = {}
    for item in items:
        item_name = item.get("item_name", "")
        cursor.execute("SELECT price FROM inventory WHERE item_name = ?", (item_name,))
        result = cursor.fetchone()
        results[item_name] = float(result[0]) if result and result[0] is not None else 0.0
    
    conn.close()
    return results

