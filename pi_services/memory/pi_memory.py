"""
Minimal Memory for Pi - Face Database Access
Supports multiple embeddings per person (different angles)
Falls back to local JSON file if DB is unreachable.
"""

import psycopg2
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_LOCAL_FACES_FILE = Path(__file__).resolve().parent / "faces_local.json"


def _load_local() -> dict:
    if _LOCAL_FACES_FILE.exists():
        try:
            return json.loads(_LOCAL_FACES_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_local(data: dict):
    try:
        _LOCAL_FACES_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"Local save error: {e}")


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', '5432'),
        sslmode='require',
        connect_timeout=5,
    )


def save_face(name: str, embedding: np.ndarray, angle: str = "front") -> bool:
    key = f"{name}__{angle}"
    embedding_list = embedding.flatten().tolist()

    # always save locally
    local = _load_local()
    local[key] = embedding_list
    _save_local(local)
    print(f"✅ Saved face locally: {key}")

    # try DB too
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        embedding_json = json.dumps(embedding_list)
        cur.execute(
            "INSERT INTO faces (name, embedding) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET embedding = %s",
            (key, embedding_json, embedding_json)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved face to DB: {key}")
    except Exception as e:
        print(f"DB save skipped (using local): {e}")

    return True


def get_all_faces() -> Dict[str, List[np.ndarray]]:
    faces: Dict[str, List[np.ndarray]] = {}

    # try DB first
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM faces")
        results = cur.fetchall()
        cur.close()
        conn.close()
        for key, embedding in results:
            try:
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                emb_array = np.array(embedding, dtype=np.float32).flatten()
                if emb_array.size == 0:
                    continue
                name = key.split("__")[0] if "__" in key else key
                if name not in faces:
                    faces[name] = []
                faces[name].append(emb_array)
            except Exception as e:
                print(f"Error loading {key}: {e}")
        print(f"✅ Loaded {len(faces)} people from DB")
        return faces
    except Exception as e:
        print(f"DB unavailable, loading from local file: {e}")

    # fallback to local file
    local = _load_local()
    for key, embedding in local.items():
        try:
            emb_array = np.array(embedding, dtype=np.float32).flatten()
            if emb_array.size == 0:
                continue
            name = key.split("__")[0] if "__" in key else key
            if name not in faces:
                faces[name] = []
            faces[name].append(emb_array)
        except Exception as e:
            print(f"Error loading local {key}: {e}")
    print(f"✅ Loaded {len(faces)} people from local file")
    return faces


def delete_person(name: str) -> int:
    count = 0

    # remove from local
    local = _load_local()
    keys_to_delete = [k for k in local if k.split("__")[0] == name]
    for k in keys_to_delete:
        del local[k]
        count += 1
    if keys_to_delete:
        _save_local(local)
        print(f"✅ Deleted {count} local embeddings for {name}")

    # try DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM faces WHERE name LIKE %s", (f"{name}%",))
        count += cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB delete skipped: {e}")

    return count
