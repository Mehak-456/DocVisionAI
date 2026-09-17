import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any
from app.configs.config import settings

logger = logging.getLogger("DocVisionAI.DB")

# Database path (stored alongside the app)
DB_PATH = os.path.join(settings.base_dir, "app", "docvisionai.db")

def get_db_connection():
    """Establishes connection to SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Documents Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            extracted_data TEXT,
            processed_image_url TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized successfully at: {DB_PATH}")

# Run schema initialization on module load
init_db()

# --- Document Repository Functions ---

def create_document(
    filename: str,
    document_type: str,
    status: str,
    extracted_data: dict,
    processed_image_url: str
) -> Dict[str, Any]:
    """Saves an extracted document record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    json_str = json.dumps(extracted_data)
    cursor.execute("""
        INSERT INTO documents (filename, document_type, status, extracted_data, processed_image_url)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, document_type, status, json_str, processed_image_url))
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return get_document_by_id(doc_id)

def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieves all documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, document_type, uploaded_at, status, extracted_data, processed_image_url
        FROM documents
        ORDER BY uploaded_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        if d.get("extracted_data"):
            try:
                d["extracted_data"] = json.loads(d["extracted_data"])
            except Exception:
                pass
        results.append(d)
    return results

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, document_type, uploaded_at, status, extracted_data, processed_image_url
        FROM documents
        WHERE id = ?
    """, (doc_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        d = dict(row)
        if d.get("extracted_data"):
            try:
                d["extracted_data"] = json.loads(d["extracted_data"])
            except Exception:
                pass
        return d
    return None

def delete_document_by_id(doc_id: int) -> bool:
    """Deletes a document entry by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0
