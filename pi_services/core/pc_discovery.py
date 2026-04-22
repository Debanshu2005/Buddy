"""
Auto-discover the Buddy PC on the local network.

Scans the subnet for a host running the STT server (port 8765).
Falls back to mDNS (buddypc.local) and then localhost.

Usage:
    from core.pc_discovery import discover_pc_ip
    ip = discover_pc_ip()
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from typing import Optional


_PROBE_PORTS = [8765, 8000]   # STT, LLM
_TIMEOUT = 0.4                # seconds per probe
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


def _tcp_reachable(host: str, port: int, timeout: float = _TIMEOUT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_local_subnet() -> Optional[str]:
    """Return the local subnet prefix e.g. '192.168.1'."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        return ".".join(parts[:3])
    except OSError:
        return None


def _scan_subnet(subnet: str) -> Optional[str]:
    """Scan all 254 hosts on the subnet concurrently."""
    found: list[str] = []
    lock = threading.Lock()

    def probe(host: str):
        for port in _PROBE_PORTS:
            if _tcp_reachable(host, port):
                with lock:
                    found.append(host)
                return

    threads = [
        threading.Thread(target=probe, args=(f"{subnet}.{i}",), daemon=True)
        for i in range(1, 255)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_TIMEOUT + 0.2)

    return found[0] if found else None


def discover_pc_ip(verbose: bool = True) -> str:
    """
    Return the IP of the Buddy PC.
    Order of attempts:
      1. BUDDY_PC_IP env var (manual override)
      2. Previously saved value in .env
      3. mDNS buddypc.local
      4. Subnet scan
      5. 127.0.0.1 (same machine)
    """
    # 1. Manual override
    override = os.getenv("BUDDY_PC_IP", "").strip()
    if override:
        if verbose:
            print(f"[Discovery] Using BUDDY_PC_IP override: {override}")
        return override

    # 2. Already in .env
    env_ip = os.getenv("BUDDY_STT_SERVER_IP", "").strip()
    if env_ip and env_ip not in ("buddypc.local", "0.0.0.0", "127.0.0.1"):
        for port in _PROBE_PORTS:
            if _tcp_reachable(env_ip, port):
                if verbose:
                    print(f"[Discovery] Using cached IP from env: {env_ip}")
                return env_ip

    # 3. mDNS
    try:
        mdns_ip = socket.gethostbyname("buddypc.local")
        for port in _PROBE_PORTS:
            if _tcp_reachable(mdns_ip, port):
                if verbose:
                    print(f"[Discovery] Found via mDNS buddypc.local → {mdns_ip}")
                _save_to_env(mdns_ip)
                return mdns_ip
    except OSError:
        pass

    # 4. Subnet scan
    if verbose:
        print("[Discovery] Scanning local network for Buddy PC...")
    subnet = _get_local_subnet()
    if subnet:
        ip = _scan_subnet(subnet)
        if ip:
            if verbose:
                print(f"[Discovery] Found Buddy PC at {ip}")
            _save_to_env(ip)
            return ip

    # 5. Localhost fallback
    if verbose:
        print("[Discovery] PC not found on network — falling back to 127.0.0.1")
    return "127.0.0.1"


def _save_to_env(ip: str):
    """Persist the discovered IP into the .env file."""
    env_path = os.path.abspath(_ENV_PATH)
    lines: list[str] = []
    keys_to_set = {
        "BUDDY_STT_SERVER_IP": ip,
        "BUDDY_PC_CAMERA_IP": ip,
        "LLM_SERVICE_URL": f"http://{ip}:8000",
    }
    updated = set()

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            key = line.split("=")[0].strip()
            if key in keys_to_set:
                new_lines.append(f"{key}={keys_to_set[key]}\n")
                updated.add(key)
            else:
                new_lines.append(line)
        lines = new_lines

    for key, val in keys_to_set.items():
        if key not in updated:
            lines.append(f"{key}={val}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[Discovery] Saved PC IP {ip} to {env_path}")
    except OSError as e:
        print(f"[Discovery] Could not save to .env: {e}")
