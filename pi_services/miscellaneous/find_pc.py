#!/usr/bin/env python3
"""Auto-discover PC running LLM service on local network"""

import socket
import requests
from concurrent.futures import ThreadPoolExecutor
import subprocess

def get_network_range():
    """Get current network range"""
    try:
        # Get default gateway
        result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'default' in line:
                gateway = line.split()[2]
                # Convert to network range (e.g., 192.168.1.1 -> 192.168.1.0/24)
                parts = gateway.split('.')
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except:
        pass
    
    # Fallback common ranges
    return ["192.168.1.0/24", "192.168.0.0/24", "10.0.0.0/24"]

def check_llm_service(ip):
    """Check if LLM service is running on this IP"""
    try:
        response = requests.get(f"http://{ip}:8000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if "Buddy" in data.get("service", ""):
                return ip
    except:
        pass
    return None

def find_llm_service():
    """Scan network for LLM service"""
    print("🔍 Scanning network for LLM service...")
    
    # Generate IP range
    ips = []
    for i in range(1, 255):
        ips.extend([
            f"192.168.1.{i}",
            f"192.168.0.{i}", 
            f"10.0.0.{i}"
        ])
    
    # Parallel scan
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(check_llm_service, ips)
    
    found_ips = [ip for ip in results if ip]
    
    if found_ips:
        print(f"✅ Found LLM service at: {found_ips[0]}")
        return f"http://{found_ips[0]}:8000"
    else:
        print("❌ LLM service not found on local network")
        return None

if __name__ == "__main__":
    service_url = find_llm_service()
    if service_url:
        print(f"Use: export LLM_SERVICE_URL='{service_url}'")
