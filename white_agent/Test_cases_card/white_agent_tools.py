"""
Tools for the White (instruction) agent personas used in the WebShop benchmark demo.

- Reads persona prompts from Test_cases_card/*.toml
- Exposes helper tools so an agent can list personas, fetch prompt text,
  and optionally broadcast the instruction to the running WebShop server.
- Mirrors the environment detection used by the green agent tools
  (WEBSHOP_REPO, WEBSHOP_PORT, etc.) so both agents talk to the same backend.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for <=3.10
    import tomli as tomllib  # type: ignore

from agentbeats import tool

SCENARIO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_CASES_DIR = SCENARIO_ROOT / "Test_cases_card"
SCENARIO_TOML = SCENARIO_ROOT / "scenario.toml"
DEFAULT_PORT = int(os.environ.get("WEBSHOP_PORT", "3000"))

PROMPT_FILES = {
    "fast_success": TEST_CASES_DIR / "white_agent_fast_success.toml",
    "exploratory": TEST_CASES_DIR / "white_agent_exploratory.toml",
    "partial_failure": TEST_CASES_DIR / "white_agent_partial_failure.toml",
    "crash": TEST_CASES_DIR / "white_agent_crash.toml",
}


@dataclass
class PersonaPrompt:
    key: str
    name: str
    description: str
    content: str


def _load_persona(key: str) -> PersonaPrompt:
    if key not in PROMPT_FILES:
        raise ValueError(f"Unknown persona '{key}'. Available: {list(PROMPT_FILES)}")
    path = PROMPT_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"Persona prompt file missing: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    agent_block = data.get("agent", {})
    prompt_block = data.get("agent.prompt", {}) or data.get("agent_prompt", {})
    # Some TOML parsers flatten dotted tables differently; guard defensively.
    if "prompt" in agent_block:
        # tomllib keeps nested tables as dicts; convert if present.
        prompt_dict = agent_block["prompt"]
    else:
        prompt_dict = data.get("agent.prompt", {})
    content = None
    if isinstance(prompt_dict, dict):
        content = prompt_dict.get("content")
    if content is None and "content" in data:
        content = data["content"]
    if content is None:
        raise ValueError(f"Prompt content missing in {path}")
    return PersonaPrompt(
        key=key,
        name=data.get("name", key.replace("_", " ").title()),
        description=data.get("description", agent_block.get("style", "")),
        content=content.strip(),
    )


def _scenario_metadata() -> Dict[str, object]:
    if not SCENARIO_TOML.exists():
        return {}
    data = tomllib.loads(SCENARIO_TOML.read_text(encoding="utf-8"))
    scenario = data.get("scenario", {})
    agents = data.get("agents", [])
    return {
        "name": scenario.get("name"),
        "description": scenario.get("description"),
        "version": scenario.get("version"),
        "agents": [a.get("name") for a in agents],
    }


def _http_post(url: str, payload: Dict[str, object], timeout: float = 5.0) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "agentbeats-white-agent"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                parsed = json.loads(raw.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                parsed = {"raw": raw.decode("utf-8", errors="ignore")}
            return {
                "status": resp.status,
                "reason": resp.reason,
                "headers": dict(resp.headers),
                "body": parsed,
            }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        return {"status": e.code, "error": e.reason, "body": body}
    except Exception as exc:  # pragma: no cover - network issue
        return {"status": None, "error": str(exc)}


def _webshop_base_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


@tool
def list_white_personas() -> Dict[str, object]:
    """
    Returns the available White agent personas defined in Test_cases_card.
    """
    personas: List[Dict[str, str]] = []
    for key in PROMPT_FILES:
        persona = _load_persona(key)
        personas.append(
            {
                "key": persona.key,
                "name": persona.name,
                "description": persona.description,
            }
        )
    return {
        "personas": personas,
        "scenario": _scenario_metadata(),
    }


@tool
def fetch_white_instruction(persona_key: str) -> Dict[str, str]:
    """
    Loads the natural-language instruction template for a given persona.
    """
    persona = _load_persona(persona_key)
    return {
        "key": persona.key,
        "name": persona.name,
        "description": persona.description,
        "instruction": persona.content,
    }


@tool
def broadcast_white_instruction(
    persona_key: str,
    conversation_id: str,
    port: int = DEFAULT_PORT,
    endpoint: str = "/api/white/instruction",
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Sends the persona instruction to a running WebShop server via HTTP POST.

    Args:
        persona_key: fast_success | exploratory | partial_failure | crash
        conversation_id: arbitrary identifier used by the caller to correlate the instruction
        port: WebShop server port (defaults to WEBSHOP_PORT env or 3000)
        endpoint: relative endpoint that accepts the instruction payload
        metadata: optional extra fields added to the payload (e.g., case, seed)
    """
    persona = _load_persona(persona_key)
    payload: Dict[str, object] = {
        "conversation_id": conversation_id,
        "persona": persona.key,
        "instruction": persona.content,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if metadata:
        payload["metadata"] = metadata

    base_url = _webshop_base_url(port)
    url = f"{base_url}{endpoint}"
    response = _http_post(url, payload)
    return {
        "request": {"url": url, "payload": payload},
        "response": response,
    }


@tool
def scenario_overview() -> Dict[str, object]:
    """
    Returns the name/description/version from scenario.toml plus agent names.
    """
    meta = _scenario_metadata()
    if not meta:
        raise FileNotFoundError(f"scenario.toml not found at {SCENARIO_TOML}")
    return meta


__all__ = [
    "list_white_personas",
    "fetch_white_instruction",
    "broadcast_white_instruction",
    "scenario_overview",
]
