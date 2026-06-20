import asyncio, json, sys
from websockets.asyncio.server import serve

CLIENTS = set()

async def relay(sender, obj):
    targets = [c for c in CLIENTS if c is not sender]
    if not targets:
        print("[ws] (no other client to relay to — is the browser open?)")
        return
    data = json.dumps(obj)
    await asyncio.gather(*(c.send(data) for c in targets), return_exceptions=True)

async def handler(ws):
    CLIENTS.add(ws)
    print(f"[ws] client linked ({len(CLIENTS)} total)")
    await ws.send(json.dumps({"type": "hello", "msg": "linked"}))
    try:
        async for msg in ws:
            try:
                obj = json.loads(msg)
            except Exception:
                print(f"[ws] non-json: {msg}"); continue
            if obj.get("type") in ("speak", "emotion"):
                print(f"[ws] relay {obj['type']}: {obj.get('audio','')} {obj.get('emotion','')}")
                await relay(ws, obj)        # forward to the browser
            else:
                print(f"[ws] from client: {msg}")
    finally:
        CLIENTS.discard(ws)
        print(f"[ws] client left ({len(CLIENTS)} total)")

async def broadcast(obj):                    # used by the stdin tester below
    if not CLIENTS:
        print("[ws] no client connected"); return
    data = json.dumps(obj)
    await asyncio.gather(*(c.send(data) for c in CLIENTS), return_exceptions=True)

async def stdin_loop():
    print("[ws] manual test:  <audiofile> [emotion]   e.g.  aiko_test_0.wav happy")
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line: break
        parts = line.split()
        if not parts: continue
        audio = parts[0]; emotion = parts[1] if len(parts) > 1 else "neutral"
        await broadcast({"type": "speak", "audio": audio, "emotion": emotion})
        print(f"[ws] sent speak: {audio} ({emotion})")

async def main():
    async with serve(handler, "localhost", 8765):
        print("[ws] listening on ws://localhost:8765"); await stdin_loop()

asyncio.run(main())