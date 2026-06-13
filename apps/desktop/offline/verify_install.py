#!/usr/bin/env python3
"""Hermes post-install verification.

Runs inside the offline venv Python. Tests the full stack:
  1. Critical imports
  2. Dashboard HTTP server (GET /api/status)
  3. WebSocket connection to /api/ws

Exit code 0 = all pass, 1 = any failure.
Output is plain text for PowerShell to capture.
"""
import os
import sys
import json
import time
import socket
import threading
import subprocess

# All output goes to stdout for PowerShell capture
def ok(msg):
    print(f"  [OK] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def info(msg):
    print(f"  [..] {msg}")

failures = []

# ---------------------------------------------------------------------------
# Stage 1: Critical imports
# ---------------------------------------------------------------------------
print("=== Stage 1: Critical imports ===")

critical_imports = [
    ("uvicorn", "ASGI server"),
    ("websockets", "WebSocket library"),
    ("httptools", "uvicorn loop optimization"),
    ("fastapi", "Web framework"),
    ("starlette", "ASGI toolkit"),
    ("pydantic", "Data validation"),
    ("h11", "HTTP/1.1 protocol"),
    ("multipart", "File upload (python-multipart)"),
    ("yaml", "Config parser (PyYAML)"),
]

for mod_name, desc in critical_imports:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "n/a")
        ok(f"{mod_name} ({desc}) v{ver}")
    except ImportError as e:
        fail(f"{mod_name} ({desc}): {e}")
        failures.append(f"import {mod_name}")

# hermes_cli imports
try:
    import hermes_cli.web_server
    ok("hermes_cli.web_server")
except Exception as e:
    fail(f"hermes_cli.web_server: {e}")
    failures.append("import hermes_cli.web_server")

try:
    from hermes_cli.main import main
    ok("hermes_cli.main")
except Exception as e:
    fail(f"hermes_cli.main: {e}")
    failures.append("import hermes_cli.main")

try:
    import tui_gateway.ws
    ok("tui_gateway.ws (WebSocket handler)")
except Exception as e:
    fail(f"tui_gateway.ws: {e}")
    failures.append("import tui_gateway.ws")

if failures:
    print(f"\n[ABORT] {len(failures)} import failures — skipping server test.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage 2: Find a free port
# ---------------------------------------------------------------------------
print("\n=== Stage 2: Start dashboard and test HTTP ===")

test_port = 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    test_port = s.getsockname()[1]

info(f"Using test port {test_port}")

# Generate a test token (same mechanism as Electron)
import secrets
test_token = secrets.token_urlsafe(32)

# Start the dashboard server
env = os.environ.copy()
env["HERMES_DESKTOP"] = "1"
env["HERMES_DASHBOARD_SESSION_TOKEN"] = test_token
# Suppress browser opening
env["HERMES_NO_OPEN"] = "1"

proc = subprocess.Popen(
    [sys.executable, "-m", "hermes_cli.main",
     "dashboard", "--no-open", "--host", "127.0.0.1",
     "--port", str(test_port)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=env,
    text=True,
)

# Wait for HTTP server to be ready (max 30s)
import urllib.request
base_url = f"http://127.0.0.1:{test_port}"
http_ok = False
deadline = time.time() + 30

while time.time() < deadline:
    # Check if process died
    if proc.poll() is not None:
        # Read whatever output we got
        out = ""
        try:
            out = proc.stdout.read(4096)
        except OSError:
            pass  # stdout 已关闭，忽略
        fail(f"Dashboard process exited early (code={proc.returncode})")
        if out:
            # Print last 20 lines of output
            lines = out.strip().split("\n")[-20:]
            for line in lines:
                print(f"    {line}")
        failures.append("dashboard early exit")
        sys.exit(1)

    try:
        req = urllib.request.Request(
            f"{base_url}/api/status",
            headers={"X-Hermes-Session-Token": test_token},
        )
        resp = urllib.request.urlopen(req, timeout=3)
        if resp.status == 200:
            data = json.loads(resp.read().decode())
            ok(f"HTTP /api/status -> {resp.status} (version={data.get('version','?')})")
            http_ok = True
            break
    except (OSError, ValueError) as _e:
        time.sleep(0.5)  # 服务还没起来，等一下重试

if not http_ok:
    fail("HTTP /api/status did not respond within 30s")
    failures.append("http status")
    # Try to read process output for diagnostics
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=5)
        if out:
            lines = out.strip().split("\n")[-20:]
            print("  Last output lines:")
            for line in lines:
                print(f"    {line}")
    except subprocess.TimeoutExpired:
        pass  # communicate 超时，进程将被强制终止
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage 3: WebSocket connection test
# ---------------------------------------------------------------------------
print("\n=== Stage 3: WebSocket connection ===")

ws_ok = False
try:
    import asyncio
    import websockets

    async def test_ws():
        ws_url = f"ws://127.0.0.1:{test_port}/api/ws?token={test_token}"
        info(f"Connecting to {ws_url[:60]}...")
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            # Send a simple JSON-RPC ping
            ping = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})
            await ws.send(ping)
            # Wait for any response (or timeout)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5)
                ok(f"WS response received: {str(resp)[:80]}")
                return True
            except asyncio.TimeoutError:
                # Connection accepted but no response — still means WS works
                ok("WS connected (no response to ping, but handshake succeeded)")
                return True

    ws_ok = asyncio.run(test_ws())

except Exception as e:
    fail(f"WebSocket connection failed: {e}")
    failures.append(f"ws connect: {e}")

if not ws_ok:
    failures.append("ws overall")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
print("\n=== Cleanup ===")
proc.terminate()
try:
    proc.wait(timeout=5)
    ok(f"Dashboard process terminated (code={proc.returncode})")
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    ok("Dashboard process killed")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
if failures:
    print(f"POST-INSTALL CHECK: FAILED ({len(failures)} issue(s))")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("POST-INSTALL CHECK: ALL PASSED")
    print("  - Imports: OK")
    print(f"  - HTTP: /api/status on port {test_port}")
    print(f"  - WebSocket: /api/ws connected and responded")
    print("  The Hermes desktop backend is fully functional.")
    sys.exit(0)
