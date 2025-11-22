#!/usr/bin/env python3
"""Lightweight compatibility server for the WebShop green agent.

Provides:
 - GET /.well-known/agent-card.json
 - GET /status
 - GET /tools
 - POST /jsonrpc      (JSON-RPC 2.0 compatible)
 - POST /v1/tool/call (compat REST shape)

This file is written to be executable directly with `python` and to be
used as a small FastAPI app. It is intentionally defensive about
parameter shapes and aims to be backward compatible for orchestrators
that send different shapes for remote tool invocation.
"""

from typing import Any, Dict, List, Optional
import os
import socket
import traceback
from pathlib import Path
import importlib.util

from fastapi import FastAPI, Body, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

APP = FastAPI()

ROOT = Path(__file__).parent
GREEN_AGENT_PORT = int(os.environ.get("GREEN_AGENT_PORT", os.environ.get("PORT", 9031)))
HOST = os.environ.get("GREEN_AGENT_HOST", "127.0.0.1")
WEBSHOP_RPC_COMPAT = os.environ.get("WEBSHOP_RPC_COMPAT", "1") in ("1", "true", "True")
WEBSHOP_AUTO_SETUP = os.environ.get("WEBSHOP_AUTO_SETUP", "0") in ("1", "true", "True")

TOOLS_PATH = ROOT / "tools.py"


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


def load_tools_module():
    # Load tools.py directly by path to avoid relying on package import and PYTHONPATH
    if not TOOLS_PATH.exists():
        raise FileNotFoundError(f"tools.py not found at {TOOLS_PATH}")
    spec = importlib.util.spec_from_file_location("green_tools", str(TOOLS_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module


def get_tools_list() -> List[Dict[str, Any]]:
    try:
        tools_mod = load_tools_module()
    except Exception as e:
        return [{"name": "<tools import failed>", "error": str(e)}]

    tool_names = []
    for name in dir(tools_mod):
        if name.startswith("_"):
            continue
        attr = getattr(tools_mod, name)
        if callable(attr):
            tool_names.append({"name": name, "doc": (attr.__doc__ or "").strip()})
    return tool_names


def run_tool(name: str, args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None):
    args = args or []
    kwargs = kwargs or {}
    tools_mod = load_tools_module()
    if not hasattr(tools_mod, name):
        raise AttributeError(f"tool '{name}' not found in green_agent.tools")
    func = getattr(tools_mod, name)
    return func(*args, **kwargs)


@APP.get("/.well-known/agent-card.json")
async def agent_card():
    url = f"http://{HOST}:{GREEN_AGENT_PORT}/"
    card = {
        "name": "webshop-green-agent",
        "description": "Green agent for WebShop benchmark (compat shim)",
        "url": url,
        "capabilities": ["tools", "jsonrpc"],
    }
    return JSONResponse(card)


@APP.get("/status")
async def status():
    return {"status": "ok", "port": GREEN_AGENT_PORT}


@APP.get("/tools")
async def tools_list():
    return {"tools": get_tools_list()}


def make_jsonrpc_response(result: Any = None, id: Any = None, error: Any = None):
    if error is not None:
        return {"jsonrpc": "2.0", "error": error, "id": id}
    return {"jsonrpc": "2.0", "result": result, "id": id}


@APP.post("/jsonrpc")
@APP.post("/")
async def jsonrpc_endpoint(payload: Dict = Body(...)):
    # Accept either a JSON-RPC request or a plain object with method/params
    try:
        jsonrpc = payload.get("jsonrpc")
        if jsonrpc == "2.0":
            method = payload.get("method")
            params = payload.get("params", {})
            req_id = payload.get("id")
        else:
            # Accept legacy shapes: {"method":"agent.get_tools"} or {"name":..., "arguments":...}
            method = payload.get("method") or payload.get("name") or payload.get("tool")
            params = payload.get("params") or payload.get("arguments") or {}
            req_id = payload.get("id")

        if method in ("agent.get_tools", "agent.tools", "get_tools"):
            return make_jsonrpc_response(get_tools_list(), id=req_id)

        if method in ("agent.tool_call", "tool_call", "tool.call"):
            # params may be {"name":..., "arguments":{...}} or positional [name, args]
            if isinstance(params, dict):
                name = params.get("name") or params.get("tool")
                arguments = params.get("arguments") or params.get("kwargs") or params.get("params") or {}
                args = []
                kwargs = arguments if isinstance(arguments, dict) else {}
            elif isinstance(params, list) and len(params) >= 1:
                name = params[0]
                args = params[1] if len(params) > 1 and isinstance(params[1], list) else []
                kwargs = params[2] if len(params) > 2 and isinstance(params[2], dict) else {}
            else:
                raise ValueError("Unable to parse params for tool call")

            res = run_tool(name, args=args, kwargs=kwargs)
            return make_jsonrpc_response(res, id=req_id)

        # unknown method
        return make_jsonrpc_response(None, id=req_id, error={"code": -32601, "message": f"Method '{method}' not found"})
    except Exception as e:
        tb = traceback.format_exc()
        return make_jsonrpc_response(None, id=payload.get("id"), error={"code": -32000, "message": str(e), "data": tb})


@APP.post("/v1/tool/call")
async def v1_tool_call(body: Dict = Body(...)):
    """Compatibility REST endpoint.

    Accepts several shapes, for example:
    - {"name":"setup_env","arguments":{"mode":"dev","port":3004}}
    - {"tool":"setup_env","arguments":{...}}
    - {"name":"setup_env","args":[...]} (positional)
    """
    try:
        name = body.get("name") or body.get("tool")
        if not name and isinstance(body.get("arguments"), dict) and len(body) == 1:
            # maybe passed as {"setup_env": { ... }}
            name, body = next(iter(body.items()))

        if not name:
            raise HTTPException(status_code=400, detail="tool name missing")

        # gather args/kwargs
        if "arguments" in body:
            arguments = body.get("arguments") or {}
            args = []
            kwargs = arguments if isinstance(arguments, dict) else {}
        elif "args" in body or "arguments" in body:
            args = body.get("args") or []
            kwargs = body.get("kwargs") or {}
        else:
            args = body.get("args") or []
            kwargs = body.get("kwargs") or {}

        # If the user asked to setup the env, perform a port-check for idempotency
        if name == "setup_env" and isinstance(kwargs, dict):
            port = kwargs.get("port") or (args[0] if args else None)
            if port and is_port_open(HOST, int(port)):
                return {"result": {"status": "already_running", "port": int(port)}}

        result = run_tool(name, args=args, kwargs=kwargs)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})


@APP.on_event("startup")
async def _on_startup():
    print(f"🟢 Green Agent Server starting on {HOST}:{GREEN_AGENT_PORT} (compat={WEBSHOP_RPC_COMPAT})")


@APP.on_event("shutdown")
async def _on_shutdown():
    print("🟡 Green Agent Server shutting down...")


if __name__ == "__main__":
    port = int(os.environ.get("GREEN_AGENT_PORT", os.environ.get("PORT", GREEN_AGENT_PORT)))
    # Pass the APP object directly so uvicorn does not re-import module paths
    uvicorn.run(APP, host=HOST, port=port, log_level="info")
