content = open('buddy_phone_link.py', encoding='utf-8').read()

# Move _startup_greeting call to run in background thread
# so the camera loop starts immediately without waiting for greeting to finish
old = (
    '        # Start phone notification listener in background\n'
    '        self._start_phone_listener()\n'
    '\n'
    '        # Initial greeting\n'
    '        self._startup_greeting()\n'
)
new = (
    '        # Start phone notification listener in background\n'
    '        self._start_phone_listener()\n'
    '\n'
    '        # Greeting runs in background so camera loop starts immediately\n'
    '        threading.Thread(target=self._startup_greeting, daemon=True).start()\n'
)

# Also remove the sleep(1.0) at the start of _startup_greeting
old2 = (
    '    def _startup_greeting(self):\n'
    '        """Detect and greet person on startup with robust recognition"""\n'
    '        try:\n'
    '            print("Starting camera and face recognition...")\n'
    '            time.sleep(1.0)\n'
)
new2 = (
    '    def _startup_greeting(self):\n'
    '        """Detect and greet person on startup with robust recognition"""\n'
    '        try:\n'
    '            print("Starting camera and face recognition...")\n'
    '            # small wait for camera to warm up, non-blocking since we run in a thread\n'
    '            time.sleep(0.5)\n'
)

if old not in content:
    print('FAIL: startup greeting anchor not found')
elif old2 not in content:
    print('FAIL: sleep(1.0) anchor not found')
else:
    content = content.replace(old, new)
    content = content.replace(old2, new2)
    open('buddy_phone_link.py', 'w', encoding='utf-8').write(content)
    print('OK: startup greeting moved to background thread')
    print('OK: startup sleep reduced from 1.0s to 0.5s')
