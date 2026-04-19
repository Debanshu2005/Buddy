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
_LOCAL_PASSWORDS_FILE = Path(__file__).resolve().parent / "passwords_local.json"
_PHOTOS_DIR = Path(__file__).resolve().parent / "face_photos"
_PHOTOS_DIR.mkdir(exist_ok=True)


def save_face_photo(name: str, image_bgr) -> bool:
    """Save a BGR numpy face photo as JPEG for this person."""
    import cv2
    path = _PHOTOS_DIR / f"{name.lower()}.jpg"
    ok = cv2.imwrite(str(path), image_bgr)
    if ok:
        print(f"\u2705 Face photo saved: {path}")
    return ok


def load_face_photo(name: str):
    """Load stored face photo for name. Returns BGR numpy array or None."""
    import cv2
    path = _PHOTOS_DIR / f"{name.lower()}.jpg"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def compare_face_photos(img_a, img_b, threshold: float = 0.4) -> tuple:
    """
    Compare two BGR face images using histogram correlation.
    Returns (match: bool, score: float). Score 1.0 = identical, 0.0 = completely different.
    threshold=0.4 means 40% similarity required — loose enough for lighting changes.
    """
    import cv2
    import numpy as np
    scores = []
    for img in (img_a, img_b):
        if img is None:
            return False, 0.0
    for i in range(3):  # B, G, R channels
        h_a = cv2.calcHist([img_a], [i], None, [64], [0, 256])
        h_b = cv2.calcHist([img_b], [i], None, [64], [0, 256])
        cv2.normalize(h_a, h_a)
        cv2.normalize(h_b, h_b)
        scores.append(cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL))
    score = float(np.mean(scores))
    return score >= threshold, round(score, 3)


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


def _load_passwords() -> dict:
    """Load passwords from local file (mirrors what's in DB)."""
    if _LOCAL_PASSWORDS_FILE.exists():
        try:
            return json.loads(_LOCAL_PASSWORDS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_passwords_local(data: dict):
    try:
        _LOCAL_PASSWORDS_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"Password local save error: {e}")


def save_password(name: str, password: str) -> bool:
    """Save password in DB faces table as name__password key, with local fallback."""
    key = f"{name}__password"
    pw = password.lower().strip()

    # save locally always
    local = _load_passwords()
    local[name.lower()] = pw
    _save_passwords_local(local)
    print(f"\u2705 Password saved locally for {name}")

    # save to DB in faces table using password as a JSON string
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO faces (name, embedding) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET embedding = %s",
            (key, json.dumps(pw), json.dumps(pw))
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"\u2705 Password saved to DB for {name}")
    except Exception as e:
        print(f"DB password save skipped (using local): {e}")

    return True


def verify_password(name: str, password: str) -> bool:
    """Check password against DB first, then local fallback."""
    pw = password.lower().strip()
    key = f"{name}__password"

    # try DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM faces WHERE name = %s", (key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            stored = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            return stored == pw
    except Exception:
        pass

    # fallback to local
    local = _load_passwords()
    return local.get(name.lower()) == pw


def find_name_by_password(password: str) -> Optional[str]:
    """Find name by password — checks DB first, then local."""
    pw = password.lower().strip()

    # try DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM faces WHERE name LIKE %s", ("%__password",))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for (key,) in rows:
            stored_raw = None
            try:
                conn2 = get_db_connection()
                cur2 = conn2.cursor()
                cur2.execute("SELECT embedding FROM faces WHERE name = %s", (key,))
                r = cur2.fetchone()
                cur2.close()
                conn2.close()
                if r:
                    stored_raw = json.loads(r[0]) if isinstance(r[0], str) else r[0]
            except Exception:
                pass
            if stored_raw == pw:
                return key.replace("__password", "").title()
    except Exception:
        pass

    # fallback to local
    local = _load_passwords()
    for name, stored in local.items():
        if stored == pw:
            return name.title()
    return None


def get_all_names_with_passwords() -> list:
    """Return all names that have a password registered."""
    names = set()

    # try DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM faces WHERE name LIKE %s", ("%__password",))
        for (key,) in cur.fetchall():
            names.add(key.replace("__password", ""))
        cur.close()
        conn.close()
    except Exception:
        pass

    # merge local
    for name in _load_passwords():
        names.add(name)

    return list(names)


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
