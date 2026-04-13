#!/usr/bin/env python3
"""
Simple keyboard input fallback for Buddy Pi
"""

def keyboard_input():
    """Simple keyboard input with timeout"""
    try:
        print("💬 Type your message: ", end='', flush=True)
        text = input().strip()
        if text:
            print(f"💬 You typed: '{text}'")
            return text
        return ""
    except:
        return ""

if __name__ == "__main__":
    while True:
        result = keyboard_input()
        if result:
            print(f"Got: {result}")
        if result.lower() in ['quit', 'exit']:
            break