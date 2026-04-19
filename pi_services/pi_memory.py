"""
Minimal Memory for Pi - Face Database Access
Supports multiple embeddings per person (different angles)
"""

import psycopg2
import numpy as np
import os
import json
from typing import Optional, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', '5432'),
        sslmode='require'
    )


def save_face(name: str, embedding: np.ndarray, angle: str = "front") -> bool:
    """
    Save a face embedding with an angle tag.
    Key is name__angle (e.g. Tautik__front, Tautik__left)
    Uses upsert so re-registering overwrites cleanly.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        key = f"{name}__{angle}"
        embedding_json = json.dumps(embedding.flatten().tolist())

        cur.execute(
            "INSERT INTO faces (name, embedding) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding",
            (key, embedding_json)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved face: {key}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save face {name}/{angle}: {e}")
        return False


def get_all_faces() -> Dict[str, List[np.ndarray]]:
    """
    Returns dict of {name: [embedding1, embedding2, ...]}
    Handles both old format (name) and new format (name__angle)
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM faces")
        results = cur.fetchall()
        cur.close()
        conn.close()

        faces: Dict[str, List[np.ndarray]] = {}

        for key, embedding in results:
            try:
                # Parse embedding
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                emb_array = np.array(embedding, dtype=np.float32).flatten()
                if emb_array.size == 0:
                    continue

                # Extract real name (strip __angle suffix if present)
                name = key.split("__")[0] if "__" in key else key

                if name not in faces:
                    faces[name] = []
                faces[name].append(emb_array)

            except Exception as e:
                print(f"Error loading {key}: {e}")
                continue

        print(f"✅ Loaded {len(faces)} people from DB")
        return faces

    except Exception as e:
        print(f"Database connection failed: {e}")
        return {}


def delete_person(name: str) -> int:
    """Delete all embeddings for a person"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM faces WHERE name LIKE %s", (f"{name}%",))
        count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Deleted {count} embeddings for {name}")
        return count
    except Exception as e:
        print(f"ERROR deleting {name}: {e}")
        return 0
