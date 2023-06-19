from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
import base64
import requests
from http import HTTPStatus
import asyncio

import msgpack  # type: ignore
app = FastAPI()


@app.get("/")
async def get():
    # return HTMLResponse(html)
    return FileResponse("./index.html", media_type="text/html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_bytes()
    print(len(base64.b64encode(data)))
    t = asyncio.create_task(heartbeat(websocket))
    res = await inference(data)
    await websocket.send_text(f"Message text was: {res}")
    t.cancel()
    await websocket.close()

import aiohttp
async def inference(data):
    req = {
        "binary": data,
        "id": "1",
    }
    session = aiohttp.ClientSession()
    async with session.post("http://localhost:8080/inference", data=msgpack.packb(req)) as resp:
        print(resp.status)
        res = await resp.text()
    print(res)
    await session.close()
    return res
    
async def heartbeat(ws):
    while True:
        await asyncio.sleep(2)
        await ws.send_text(f"Waiting...")
