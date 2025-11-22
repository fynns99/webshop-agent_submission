# scenarios/webshop_benchmark/green_agent/tools.py
"""
Tools for the green agent in the webshop benchmark scenario.
- Starts the WebShop from a separate repo (Flask) via run_dev.sh / run_prod.sh
- Probes reachability (no /health endpoint -> checks root HTML)
- Loads catalog/goals from data/
- Optionally reads the latest user-session log file
"""

from __future__ import annotations
import os, sys, json, shlex, subprocess, pathlib, time, socket, urllib.request, logging, importlib.util, shutil
from typing import Dict, Any, Optional, List, Sequence
from pydantic import BaseModel, ConfigDict


def _stringify(value: Any) -> str:
    """Convert arbitrary python values into a compact string representation."""
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)[:500]
    except Exception:
        return str(value)

# Define specific return types to avoid schema issues
class RuntimeInfo(BaseModel):
    python: str
    python_version: str
    virtual_env: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class PortStatus(BaseModel):
    port: int
    listening: bool
    root_accessible: Optional[bool] = None

    model_config = ConfigDict(extra="forbid")


class SetupResult(BaseModel):
    repo: str
    mode: str
    port: int
    app_url: str
    listening: bool

    model_config = ConfigDict(extra="forbid")


class CatalogFile(BaseModel):
    path: str
    status: str
    item_count: Optional[int] = None
    file_type: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class CatalogResult(BaseModel):
    base: str
    files: List[CatalogFile]
    status: str
    available_files: List[str]
    error: Optional[str]

    model_config = ConfigDict(extra="forbid")


class GoalFile(BaseModel):
    path: str
    status: str
    item_count: Optional[int] = None
    preview: Optional[str] = None
    file_type: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class GoalsResult(BaseModel):
    goals_dir: str
    available: List[str]
    status: str
    details: List[GoalFile]

    model_config = ConfigDict(extra="forbid")


class LogResult(BaseModel):
    base: str
    found: bool
    path: Optional[str] = None
    preview_lines: Optional[List[str]] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class DetailItem(BaseModel):
    label: str
    value: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class EvidenceItem(BaseModel):
    label: str
    value: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class HealthResult(BaseModel):
    repo: str
    repo_exists: bool
    scripts_found: List[str]
    scripts_missing: List[str]
    data_root: str
    catalog_candidates: List[str]
    goals_dir: Optional[str]
    goal_files: List[str]
    log_dir: str
    runtime: RuntimeInfo
    port_status: PortStatus

    model_config = ConfigDict(extra="forbid")


class BattleEvent(BaseModel):
    timestamp: str
    battle_id: str
    summary: str
    reporter: str
    details: List[DetailItem]

    model_config = ConfigDict(extra="forbid")


class BattleReport(BaseModel):
    battle_id: str
    timestamp: str
    winner: str
    evidence: List[EvidenceItem]
    summary: str

    model_config = ConfigDict(extra="forbid")


# Rebuild models to resolve forward references
RuntimeInfo.model_rebuild()
PortStatus.model_rebuild()
SetupResult.model_rebuild()
CatalogFile.model_rebuild()
CatalogResult.model_rebuild()
GoalFile.model_rebuild()
GoalsResult.model_rebuild()
LogResult.model_rebuild()
DetailItem.model_rebuild()
EvidenceItem.model_rebuild()
HealthResult.model_rebuild()
BattleEvent.model_rebuild()
BattleReport.model_rebuild()

from agentbeats import tool


# --- New Gym-style environment tool functions ---
@tool
def start_env(instruction: str = "") -> dict:
    """
    Start a new WebShop Gym environment episode with a natural language instruction.

    This is the Gym-style environment used for evaluation (separate from the Flask server).
    Returns the initial observation after reset.
    """
    return start_environment(instruction)


@tool
def reset_env(instruction: str = "") -> dict:
    """
    Reset the WebShop Gym environment for a new episode.

    If the environment does not yet exist, it is created first. Returns the new observation.
    """
    return reset_environment(instruction)


@tool
def step_env(action: str) -> dict:
    """
    Take a single step in the WebShop Gym environment.

    The `action` should be a WebShop action string, e.g.:
      - "search[blue cotton hoodie under $50]"
      - "click[3]"
      - "add_to_cart[sku_123]"

    Returns the next observation, reward, done-flag, and info dict
    as a JSON-serializable dict.
    """
    global _env
    if _env is None:
        # Lazily initialize with an empty instruction if not yet started
        _env = WebShopEnv()
        obs = _env.reset("")
        return {
            "ok": True,
            "observation": obs,
            "reward": 0.0,
            "done": False,
            "info": {"note": "env was auto-initialized"},
        }

    try:
        obs, reward, done, info = _env.step(action)
    except TypeError:
        # Fallback in case the underlying env expects a different signature
        obs, reward, done, info = _env.step(action, {})  # type: ignore[misc]

    return {
        "ok": True,
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


@tool
def close_env() -> dict:
    """Close the current WebShop Gym environment, if any, and release resources."""
    global _env
    status = "no_env"
    try:
        if _env is not None:
            close_fn = getattr(_env, "close", None)
            if callable(close_fn):
                close_fn()
            status = "closed"
    finally:
        _env = None

    return {"status": status}

REPO = pathlib.Path(os.environ.get("WEBSHOP_REPO", "/Users/fynnschafer/Pycharm/Agentic_AI/WebShop"))

# Extend sys.path so the WebShop repo modules are discoverable.
for p in (REPO, REPO / "baseline_models"):
    if p.exists():
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

try:
    from baseline_models.env import WebEnv as WebShopEnv  # noqa: E402
except Exception:
    env_py = REPO / "baseline_models" / "env.py"
    if not env_py.exists():
        raise ModuleNotFoundError(f"WebShop env.py not found at {env_py}; set WEBSHOP_REPO correctly.")
    spec = importlib.util.spec_from_file_location("webshop_env", env_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    WebShopEnv = getattr(mod, "WebShopEnv", None) or getattr(mod, "WebEnv", None)
    if WebShopEnv is None:
        raise ImportError(f"'WebShopEnv' not found in {env_py}")

# Local WebShop repo – override via WEBSHOP_REPO if needed
DEFAULT_WEBSHOP_REPO = pathlib.Path("/opt/webshop").expanduser().resolve()
SCENARIO_ROOT = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

# --- Catalog paths (JSONL-first) ---
# Default collections dir and JSONL catalog; both can be overridden via env:
#   WEBSHOP_COLLECTIONS  -> collections directory (e.g., .../data/collections_1k)
#   WEBSHOP_PRODUCTS     -> explicit path to products.jsonl
DEFAULT_COLLECTIONS_DIR = os.environ.get("WEBSHOP_COLLECTIONS")
if DEFAULT_COLLECTIONS_DIR:
    COLLECTIONS_DIR = pathlib.Path(DEFAULT_COLLECTIONS_DIR).expanduser().resolve()
else:
    # _repo_root() is defined later in this module; avoid calling it at import time
    # to prevent NameError during agent startup. Use the DEFAULT_WEBSHOP_REPO
    # fallback here and let _repo_root() still provide an override at runtime
    # if needed.
    COLLECTIONS_DIR = (DEFAULT_WEBSHOP_REPO / "data" / "collections_1k")

DEFAULT_PRODUCTS_PATH = os.environ.get("WEBSHOP_PRODUCTS")
if DEFAULT_PRODUCTS_PATH:
    PRODUCTS_JSONL = pathlib.Path(DEFAULT_PRODUCTS_PATH).expanduser().resolve()
else:
    PRODUCTS_JSONL = (COLLECTIONS_DIR / "products.jsonl")

# Default port: Flask usually runs on 0.0.0.0:3000 (run_dev.sh → python -m web_agent_site.app --log --attrs)
DEFAULT_PORT = int(os.environ.get("WEBSHOP_PORT", "3000"))

_env: Optional[WebShopEnv] = None


def start_environment(instruction: str = ""):
    """Initialize the WebShop environment and start an episode."""
    global _env
    _env = WebShopEnv()
    obs = _env.reset(instruction)
    return {"ok": True, "observation": obs}


def reset_environment(instruction: str = ""):
    """Reset the environment for a new episode."""
    global _env
    if _env is None:
        _env = WebShopEnv()
    obs = _env.reset(instruction)
    return {"ok": True, "observation": obs}


def _repo_root() -> pathlib.Path:
    """Resolve the WebShop repository path with env override and sensible fallback."""
    repo_override = os.environ.get("WEBSHOP_REPO")
    repo_path = pathlib.Path(repo_override).expanduser() if repo_override else DEFAULT_WEBSHOP_REPO
    return repo_path.resolve()

def _webshop_python() -> pathlib.Path:
    """
    Prefer an explicitly configured WebShop interpreter (e.g., Py3.10 venv).
    Falls back to python3.10 in PATH or sys.executable.
    """
    candidates = [
        os.environ.get("WEBSHOP_PY"),
        shutil.which("python3.10"),
        sys.executable,
    ]
    for cand in candidates:
        if cand:
            path = pathlib.Path(cand).expanduser().resolve()
            if path.exists():
                return path
    raise FileNotFoundError("No suitable interpreter for WebShop found (WEBSHOP_PY/python3.10/sys.executable).")


def _run_async(
    cmd: Sequence[str] | str,
    cwd: pathlib.Path | None = None,
    extra_env: Dict[str, str] | None = None,
    log_path: pathlib.Path | None = None,
) -> subprocess.Popen:
    """Start a background process and optionally tee output to log_path."""
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    if isinstance(cmd, str):
        popen_args = shlex.split(cmd)
    else:
        popen_args = list(cmd)
    logger.debug("Starting background command: %s (cwd=%s)", popen_args, cwd or os.getcwd())
    stdout = subprocess.PIPE
    stderr = subprocess.PIPE
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        stdout = log_file
        stderr = log_file
    proc = subprocess.Popen(
        popen_args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True
    )
    if log_file is not None:
        setattr(proc, "_webshop_log_file", log_file)
    return proc

def _port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    """Einfacher TCP-Check, ob etwas auf host:port lauscht."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False

def _http_get(url: str, timeout: float = 2.0) -> Optional[bytes]:
    try:
        logger.debug("HTTP GET %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None

# -------------------------------
# Public tools for the Green Agent
# -------------------------------

@tool
def setup_env(mode: str = "dev", port: int = DEFAULT_PORT) -> SetupResult:
    """
    Start the WebShop from the separate repo.
    Expected startup scripts: run_dev.sh, run_prod.sh
    - dev:   python -m web_agent_site.app --log --attrs  (via run_dev.sh)
    - prod:  python -m web_agent_site.app --log          (via run_prod.sh)

    Returns basic metadata for downstream steps.
    """
    repo = _repo_root()
    if not repo.exists():
        raise FileNotFoundError(f"[tools] WebShop repo not found: {repo}")

    # Optional setup scripts if present
    setup_sh = repo / "run_dev.sh"  # dev is usually sufficient
    prod_sh  = repo / "run_prod.sh"

    env = os.environ.copy()
    env["FLASK_ENV"] = "development" if mode == "dev" else "production"
    env["PORT"] = str(port)
    env["PYTHONPATH"] = f"{str(repo)}:{str(repo / 'baseline_models')}"

    python_bin = _webshop_python()
    venv_bin = python_bin.parent
    venv_root = venv_bin.parent if venv_bin.name == "bin" else None
    if venv_root and venv_bin.exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH','')}"
        env["VIRTUAL_ENV"] = str(venv_root)

    log_path = pathlib.Path(os.environ.get("WEBSHOP_ENV_LOG", str(SCENARIO_ROOT / "logs" / "webshop_env.log"))).expanduser()
    if mode == "prod" and prod_sh.exists():
        cmd = ["/bin/bash", str(prod_sh)]
    elif setup_sh.exists():
        cmd = ["/bin/bash", str(setup_sh)]
    else:
        cmd = [str(python_bin), "-m", "web_agent_site.app", "--log"]
        if mode == "dev":
            cmd.append("--attrs")

    proc = _run_async(cmd, cwd=repo, extra_env=env, log_path=log_path)
    logger.info("Started WebShop process pid=%s via %s; logging to %s", proc.pid, cmd, log_path)

    timeout_seconds = float(os.environ.get("WEBSHOP_START_TIMEOUT", "90"))
    deadline = time.time() + timeout_seconds
    delay = 0.5
    app_url = f"http://127.0.0.1:{port}"
    listening = False
    while time.time() < deadline:
        if _port_is_listening(port):
            listening = True
            if _http_get(f"{app_url}/"):
                break
        time.sleep(delay)
        delay = min(delay * 1.5, 3.0)

    listening = listening and _port_is_listening(port)
    if not listening:
        tail = ""
        try:
            tail = "\n".join(pathlib.Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(f"WebShop did not start on port {port} within {timeout_seconds}s. Log tail:\n{tail}")

    return SetupResult(
        repo=str(repo),
        mode=mode,
        port=port,
        app_url=app_url,
        listening=listening
    )

@tool
def health(port: int = DEFAULT_PORT) -> HealthResult:
    """
    Comprehensive health check of the WebShop environment:
    - Verifies repo & scripts
    - Validates data directories and the JSONL catalog
    - Tests port status
    - Checks goal files
    """
    repo = _repo_root()
    repo_exists = repo.exists()

    # Required scripts
    script_names = ["run_web_agent_site_env.sh", "run_dev.sh", "run_prod.sh"]
    scripts_found: List[str] = []
    scripts_missing: List[str] = []
    for script in script_names:
        if (repo / script).exists():
            scripts_found.append(script)
        else:
            scripts_missing.append(script)

    # Data roots
    data_override = os.environ.get("WEBSHOP_DATA")
    data_path = pathlib.Path(data_override).expanduser() if data_override else (repo / "data")

    # Catalog candidates (JSON + JSONL)
    catalog_candidates: List[str] = []
    # Legacy JSON files (optional, for compatibility)
    legacy_candidates = [
        data_path / "items_human_ins.json",
        data_path / "items_ins_v2_1000.json",
        data_path / "items_shuffle_1000.json",
    ]
    for p in legacy_candidates:
        if p.exists():
            catalog_candidates.append(str(p))
    # Primary JSONL
    if PRODUCTS_JSONL.exists():
        catalog_candidates.append(str(PRODUCTS_JSONL))

    # Goals
    goals_dir_env = os.environ.get("WEBSHOP_GOALS")
    if goals_dir_env:
        goals_path = pathlib.Path(goals_dir_env).expanduser()
    else:
        goals_path = repo / "baseline_models" / "data"
        if not goals_path.exists():
            goals_path = repo / "data" / "goals"

    goal_files: List[str] = []
    if goals_path.exists():
        for gf in ["goal_query_map.json", "goal_query_predict.json", "human_goals.json"]:
            if (goals_path / gf).exists():
                goal_files.append(gf)

    # Logs
    log_dir = repo / "user_session_logs"
    if not log_dir.exists():
        log_dir = SCENARIO_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

    runtime = RuntimeInfo(
        python=sys.executable,
        python_version=sys.version,
        virtual_env=os.environ.get("VIRTUAL_ENV"),
    )

    # Port status
    listening = _port_is_listening(port)
    root_ok = False
    if listening:
        raw = _http_get(f"http://127.0.0.1:{port}/")
        root_ok = bool(raw)

    port_status = PortStatus(
        port=port,
        listening=listening,
        root_accessible=root_ok if listening else None,
    )

    return HealthResult(
        repo=str(repo),
        repo_exists=repo_exists,
        scripts_found=scripts_found,
        scripts_missing=scripts_missing,
        data_root=str(data_path),
        catalog_candidates=catalog_candidates,
        goals_dir=str(goals_path) if goals_path.exists() else None,
        goal_files=goal_files,
        log_dir=str(log_dir),
        runtime=runtime,
        port_status=port_status
    )

@tool
def load_catalog() -> CatalogResult:
    """
    Load catalog overview with JSONL-first priority.
    - Primary: products.jsonl (line-delimited JSON objects)
    - Legacy (optional): items_*.json (list/dict)
    """
    repo = _repo_root()
    data_override = os.environ.get("WEBSHOP_DATA")
    base = pathlib.Path(data_override).expanduser() if data_override else (repo / "data")

    if not base.exists():
        return CatalogResult(
            base=str(base),
            files=[],
            status="data directory not found",
            available_files=[],
            error=f"Directory not found: {base}"
        )

    results: List[CatalogFile] = []

    # 1) JSONL primary
    try:
        if PRODUCTS_JSONL.exists():
            count = 0
            with open(PRODUCTS_JSONL, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        count += 1
                    except Exception:
                        # Skip malformed lines but continue counting valid ones
                        continue
            results.append(CatalogFile(
                path=str(PRODUCTS_JSONL),
                status="ok" if count > 0 else "empty",
                item_count=count if count > 0 else None,
                file_type="jsonl"
            ))
        else:
            results.append(CatalogFile(
                path=str(PRODUCTS_JSONL),
                status="file not found"
            ))
    except Exception as e:
        results.append(CatalogFile(
            path=str(PRODUCTS_JSONL),
            status="error",
            error=str(e)
        ))

    # 2) Legacy JSON files (optional, shown for visibility)
    legacy_candidates = [
        base / "items_human_ins.json",
        base / "items_ins_v2_1000.json",
        base / "items_shuffle_1000.json",
    ]
    for p in legacy_candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "products" in data:
                    n = len(data["products"] or [])
                    ftype = "dict(products)"
                elif isinstance(data, list):
                    n = len(data)
                    ftype = "list"
                else:
                    n = 0
                    ftype = type(data).__name__
                results.append(CatalogFile(
                    path=str(p),
                    status="ok",
                    item_count=n,
                    file_type=ftype
                ))
            else:
                results.append(CatalogFile(
                    path=str(p),
                    status="file not found"
                ))
        except Exception as e:
            results.append(CatalogFile(
                path=str(p),
                status="error",
                error=str(e)
            ))

    ok_files = [f.path for f in results if f.status == "ok"]
    status = "ok" if any(f.status == "ok" for f in results) else "no valid files found"
    err = None if ok_files else "No catalog files found"
    return CatalogResult(
        base=str(base),
        files=results,
        status=status,
        available_files=ok_files,
        error=err
    )

# --- New tool: preview_products_jsonl ---
@tool
def preview_products_jsonl(limit: int = 5) -> Dict[str, Any]:
    """
    Preview the first N valid records from products.jsonl.
    Useful as a quick sanity check for the catalog the Green Agent should use.
    """
    out: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "path": str(PRODUCTS_JSONL),
        "exists": PRODUCTS_JSONL.exists(),
        "limit": limit,
    }
    if not PRODUCTS_JSONL.exists():
        return {"meta": meta, "records": out, "status": "file not found"}
    try:
        with open(PRODUCTS_JSONL, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(out) >= max(0, int(limit)):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out.append(rec)
                except Exception:
                    continue
        return {"meta": meta, "records": out, "status": "ok" if out else "empty"}
    except Exception as e:
        return {"meta": meta, "records": out, "status": "error", "error": str(e)}

@tool
def list_goals() -> GoalsResult:
    """List goal files with graceful handling of missing locations."""
    # Try environment variable first, then standard locations
    goals_dir = os.environ.get("WEBSHOP_GOALS")
    if goals_dir:
        base = pathlib.Path(goals_dir).expanduser()
    else:
        repo = _repo_root()
        base = repo / "baseline_models" / "data"
        if not base.exists():
            base = repo / "data" / "goals"  # alternative location

    out: List[GoalFile] = []
    goal_files = ["goal_query_map.json", "goal_query_predict.json", "human_goals.json"]
    
    if not base.exists():
        return GoalsResult(
            goals_dir=str(base),
            available=[],
            status="directory not found",
            details=[]
        )
    
    for name in goal_files:
        p = base / name
        try:
            if p.exists():
                sample = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(sample, list):
                    count = len(sample)
                    preview = _stringify(sample[:2])
                    ftype = "list"
                elif isinstance(sample, dict):
                    count = len(sample)
                    preview = _stringify(list(sample.items())[:2])
                    ftype = "dict"
                else:
                    count = 0
                    preview = _stringify(sample)
                    ftype = type(sample).__name__
                out.append(GoalFile(
                    path=str(p),
                    status="ok",
                    item_count=count,
                    preview=preview,
                    file_type=ftype
                ))
            else:
                out.append(GoalFile(
                    path=str(p),
                    status="file not found",
                    item_count=None
                ))
        except Exception as e:
            out.append(GoalFile(
                path=str(p),
                status="error",
                error=str(e)
            ))
    
    available_paths = [g.path for g in out if g.status == "ok"]

    return GoalsResult(
        goals_dir=str(base),
        available=available_paths,
        status="ok" if available_paths else "no valid goals found",
        details=out
    )

@tool
def read_latest_session_log() -> LogResult:
    """
    Read the newest log file from user_session_logs/ (written by WebShop when --log is enabled).
    If none exist yet, returns only the folder path.
    """
    repo = _repo_root()
    base = repo / "user_session_logs"
    if not base.exists():
        return LogResult(base=str(base), found=False)
    logs = sorted(base.rglob("*.jsonl")) or sorted(base.rglob("*.json"))
    if not logs:
        return LogResult(base=str(base), found=False)
    latest = logs[-1]
    try:
        # Show only the first 3 lines
        lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
        return LogResult(
            base=str(base),
            found=True,
            path=str(latest),
            preview_lines=lines
        )
    except Exception as e:
        return LogResult(
            base=str(base),
            found=True,
            path=str(latest),
            error=str(e)
        )

@tool
def update_battle_process(
    battle_id: str,
    event_summary: str,
    reporter: str = "green",
    details: Optional[Sequence[DetailItem | str]] = None,
) -> BattleEvent:
    """
    Updates the battle log with new events and observations.
    
    Args:
        battle_id: Unique identifier for the battle session
        event_summary: Brief description of what happened
        reporter: Who is reporting the event (default: green agent)
        details: Additional structured data about the event
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detail_items: List[DetailItem] = []
    if details:
        for entry in details:
            if isinstance(entry, DetailItem):
                detail_items.append(entry)
            else:
                detail_items.append(DetailItem(label=str(entry)))
    event = BattleEvent(
        timestamp=timestamp,
        battle_id=battle_id,
        summary=event_summary,
        reporter=reporter,
        details=detail_items,
    )
    
    # Log directory for battle events
    repo = _repo_root()
    log_dir = (repo / "battle_logs") if repo.exists() else (SCENARIO_ROOT / "battle_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file for this battle
    log_file = log_dir / f"battle_{battle_id}.jsonl"
    
    # Append event to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(exclude_none=True)) + "\n")
    
    return event

@tool
def report_on_battle_end(
    battle_id: str,
    winner: str,
    evidence: Optional[Sequence[EvidenceItem | str]] = None
) -> BattleReport:
    """
    Declares the battle winner and provides supporting evidence.
    
    Args:
        battle_id: Unique identifier for the battle session
        winner: Name/ID of the winning agent
        evidence: List of events/observations that justify the decision
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    evidence_items: List[EvidenceItem] = []
    if evidence:
        for entry in evidence:
            if isinstance(entry, EvidenceItem):
                evidence_items.append(entry)
            else:
                evidence_items.append(EvidenceItem(label=str(entry)))
    report = BattleReport(
        battle_id=battle_id,
        timestamp=timestamp,
        winner=winner,
        evidence=evidence_items,
        summary=f"Battle {battle_id} concluded. Winner: {winner}"
    )
    
    # Log directory for battle results
    repo = _repo_root()
    results_dir = (repo / "battle_results") if repo.exists() else (SCENARIO_ROOT / "battle_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save final report
    report_file = results_dir / f"battle_{battle_id}_result.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(exclude_none=True), f, indent=2)
        
    return report

METRIC_WEIGHTS = {
    "completion": 0.5,
    "reward": 0.3,
    "efficiency": 0.1,
    "stability": 0.1,
}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


@tool
def score_metrics(
    metrics: Dict[str, float],
    used_steps: Optional[int] = None,
    success: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compute a composite WebShop score using the current weights:
      0.5 * completion + 0.3 * reward + 0.1 * efficiency + 0.1 * stability.
    Missing metrics default to 0.0 and all inputs are clamped to [0,1].
    """
    normalized = {name: _clamp01(metrics.get(name, 0.0)) for name in METRIC_WEIGHTS}
    composite = sum(normalized[name] * weight for name, weight in METRIC_WEIGHTS.items())

    result: Dict[str, Any] = {
        "metrics": normalized,
        "composite_score": composite,
    }
    if used_steps is not None:
        result["used_steps"] = int(used_steps)
    if success is not None:
        result["success"] = bool(success)
    return result


def _derive_metrics_from_trajectory(trajectory: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight heuristic to derive metrics from a WebShop-like trajectory.
    Keeps things defensive so it can run on partial or synthetic data.
    """
    steps = 0
    success = False
    reward_vals: List[float] = []
    had_error = False

    for event in trajectory:
        if not isinstance(event, dict):
            continue
        action = event.get("action") or event.get("event")
        actor = (event.get("actor") or "").lower()
        if action or actor == "blue":
            steps += 1

        if isinstance(event.get("reward"), (int, float)):
            reward_vals.append(float(event["reward"]))

        obs = event.get("observation") or event.get("obs") or {}
        if isinstance(obs, dict):
            if obs.get("status") == "success" or obs.get("order_id"):
                success = True
            if obs.get("done") and obs.get("reward", 0) > 0:
                success = True
        info = event.get("info") or {}
        if isinstance(info, dict) and info.get("goal_met"):
            success = True

        status = (event.get("status") or "").lower()
        ok_flag = event.get("ok")
        if status == "error" or ok_flag is False or event.get("error"):
            had_error = True

    reward_score = 0.0
    if reward_vals:
        reward_score = max(reward_vals)
        reward_score = _clamp01(reward_score)
    if not reward_vals and success:
        reward_score = 1.0

    efficiency = 1.0
    if steps > 0:
        if steps <= 4:
            efficiency = 1.0
        elif steps <= 6:
            efficiency = 0.8
        elif steps <= 10:
            efficiency = 0.6
        else:
            efficiency = 0.4

    stability = 0.0 if had_error else 1.0

    metrics = {
        "completion": 1.0 if success else 0.0,
        "reward": reward_score,
        "efficiency": efficiency,
        "stability": stability,
    }
    return {"metrics": metrics, "used_steps": steps, "success": success}


@tool
def score_trajectory(trajectory: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score a WebShop trajectory dict/list using the current weighted metrics.

    The function is resilient to partial data:
    - Detects success if any observation has status=success/order_id/done+reward>0.
    - Derives steps from BLUE actions or any event with an action field.
    - Uses max reward seen; defaults to 1.0 on success if no reward provided.
    """
    derived = _derive_metrics_from_trajectory(trajectory or [])
    return score_metrics(
        metrics=derived["metrics"],
        used_steps=derived["used_steps"],
        success=derived["success"],
    )

def _load_blue_trajectory(battle_id: str) -> Dict[str, Any]:
    """
    Try to load a persisted Blue trajectory for this battle_id from common locations.
    Returns an empty structure if nothing is found.
    """
    candidates = [
        SCENARIO_ROOT / "logs" / f"{battle_id}.jsonl",
        SCENARIO_ROOT / "logs" / f"webshop_{battle_id}.jsonl",
        SCENARIO_ROOT / "logs" / f"battle_{battle_id}.jsonl",
        _repo_root() / "battle_logs" / f"battle_{battle_id}.jsonl",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            events = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue

            summary_metrics: Dict[str, Any] = {}
            success = None
            used_steps = None
            for ev in reversed(events):
                if isinstance(ev, dict) and ev.get("event_type") == "metrics":
                    summary = ev.get("summary") or {}
                    if isinstance(summary, dict):
                        summary_metrics = summary.get("metrics") or {}
                        success = summary.get("success")
                        used_steps = summary.get("used_steps")
                    break

            return {
                "events": events,
                "metrics": summary_metrics,
                "success": success,
                "used_steps": used_steps,
                "source": str(path),
            }
        except Exception:
            continue

    # Fallback: pick the latest JSONL in known dirs if nothing matched battle_id
    search_dirs = [
        SCENARIO_ROOT / "logs",
        _repo_root() / "battle_logs",
    ]
    latest_file: Optional[pathlib.Path] = None
    latest_mtime = -1.0
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.jsonl"):
            try:
                mtime = p.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = p
            except Exception:
                continue
    if latest_file:
        try:
            events = []
            for line in latest_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
            return {
                "events": events,
                "metrics": {},
                "success": None,
                "used_steps": None,
                "source": str(latest_file),
            }
        except Exception:
            pass

    return {"events": [], "metrics": {}, "source": None}


@tool
def battle_info(
    type: str,
    battle_id: str,
    agent_name: str,
    agent_id: str,
    backend_url: str,
) -> dict:
    return {
        "status": "ok",
        "battle_id": battle_id,
        "agent_id": agent_id,
        "backend_url": backend_url,
    }


@tool
def battle_start(battle_id: str) -> Dict[str, Any]:
    """
    Handle battle_start event for the Green agent (referee).
    Execute the standard Green workflow: check health, load catalog, 
    wait for Blue trajectory, score it, and report results.
    """
    import time
    
    results = []
    try:
        results.append("Starting Green referee workflow")
        results.append("health: " + str(health()))
        results.append("load_catalog: " + str(load_catalog()))
        
        # Load blue trajectory from persisted log files (JSONL)
        traj = _load_blue_trajectory(battle_id)
        results.append("get_blue_trajectory: " + str({k: v for k, v in traj.items() if k != "events"}))
        
        # Score using the configured metric weights (defaults to zeros if missing)
        if traj.get("events"):
            score_result = score_trajectory(traj["events"])
        else:
            score_result = score_metrics(
                metrics=traj.get("metrics") or {},
                used_steps=traj.get("used_steps"),
                success=traj.get("success"),
            )
        results.append("score_trajectory: " + str(score_result))
        
        # Finalize battle end report snapshot
        end_report = {"status": "completed", "battle_id": battle_id}
        results.append("report_on_battle_end: " + str(end_report))
        
        return {
            "status": "ok",
            "message": "Green workflow completed",
            "results": results,
            "battle_id": battle_id
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Green workflow failed: {e}",
            "battle_id": battle_id
        }

@tool
def message_stream(message: dict) -> dict:
    """
    Handle A2A message stream containing embedded JSON-RPC calls.
    Parse the message and execute the appropriate tool.
    """
    import json
    
    try:
        # Extract the JSON-RPC payload from the message
        if isinstance(message, dict) and 'parts' in message:
            for part in message['parts']:
                if part.get('kind') == 'text':
                    text = part.get('text', '')
                    if text.startswith('{') and '"jsonrpc"' in text:
                        # Parse the embedded JSON-RPC
                        rpc_payload = json.loads(text)
                        method = rpc_payload.get('method', '')
                        
                        if method == 'battle_info':
                            params = rpc_payload.get('params', {})
                            return battle_info(**params)
                        elif method == 'battle_start':
                            params = rpc_payload.get('params', {})
                            return battle_start(params.get('battle_id', 'unknown'))
                        else:
                            return {"error": f"Unknown method: {method}"}
        
        return {"error": "No valid JSON-RPC found in message"}
    except Exception as e:
        return {"error": f"Failed to parse message: {str(e)}"}


# Export – AgentBeats loads these functions as tools
__all__ = [
    # Web server / environment setup
    "setup_env",
    "health",
    "load_catalog",
    "preview_products_jsonl",
    "list_goals",
    "read_latest_session_log",

    # Battle logging & reports
    "update_battle_process",
    "report_on_battle_end",
    "score_metrics",
    "score_trajectory",

    # Gym-style WebShop environment helpers (internal)
    "start_environment",
    "reset_environment",

    # Gym-style WebShop environment tools (exposed to the agent)
    "start_env",
    "reset_env",
    "step_env",
    "close_env",

    # Battle / A2A integration
    "battle_info",
    "battle_start",
    "message_stream",
]
