# 1. Overview & Quickstart

This submission implements the WebShop benchmark on AgentBeats with two agents:

- **White Agents (shopping personas)**: autonomous shoppers that use a small deterministic tool server.
- **Green Agent (referee)**: hosts the WebShop text environment, verifies actions, and scores trajectories.

You can run the full benchmark in ~3 steps once the venv is active.

## 1.1 Quickstart (run one battle)

**Terminal 1 – Start AgentBeats backend**
```bash
cd <YOUR_AGENTBEATS_PATH>
source venv/bin/activate
export OPENAI_API_KEY="sk-proj-…"

agentbeats run_backend \
  --host 127.0.0.1 \
  --backend_port 9000 \
  --mcp_port 9001 \
  --reload
```

**Terminal 2 – Load scenario (auto‑launch agents in tmux)**
```bash
cd <YOUR_AGENTBEATS_PATH>
source venv/bin/activate
export OPENAI_API_KEY="sk-proj-…"

agentbeats load_scenario \
  <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final \
  --launch-mode tmux \
  --register_agents \
  --backend http://127.0.0.1:9000
```

**Terminal 3 – Run the scenario**
```bash
cd <YOUR_AGENTBEATS_PATH>
source venv/bin/activate

agentbeats run_scenario \
  <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final \
  --backend http://127.0.0.1:9000
```

Attach to tmux to watch live logs:
```bash
tmux attach -t agentbeats-webshop-benchmark
```

## 1.2 Ports used

- Backend: **9000**
- MCP: **9001**
- Green Agent: **9031**
- White Agent (current persona): **9051**

---

# 2.1 White Agent Architecture (Tools + Server)

The White Agent in this benchmark is a **fully autonomous shopping agent**. It consists of two components:

### **(1) A Tool Server (`white_agent_server.py`)**
This is a small A2A-compatible HTTP server that exposes deterministic WebShop actions to the LLM.  
It handles the following responsibilities:
- Maintain a local session ID for the WebShop env
- Forward action requests to the Green Agent or WebShop API
- Enforce allowed action types (`search[…]`, `click[…]`)
- Validate clickables and normalize input
- Return clean JSON responses for the LLM agent

The White Agent tool server exposes the following tool endpoints:
- `search(query: str)`
- `click(target: str)`
- `select(option_value: str)`
- `buy()`
- `reset_session()`

All requests are deterministic and validated, making the White Agent reproducible across runs.

### **(2) An Agent Card (`white_agent_*.toml`)**
Each persona (fast_success, exploratory, partial_failure, crash) uses:
- the same server + tool definitions  
- but different **system prompts** describing the behavior style

This allows reproducible evaluation without modifying the tool code.

---

# 2.2 Green Agent Architecture (Environment Host + Judge)

The Green Agent is the **referee** of the benchmark.  
It consists of its own A2A server (`green_agent_server.py`) which:

### **Environment Management**
- Starts the WebShop text-based environment internally  
- Loads the product catalog  
- Resets the environment for each new episode  
- Steps the environment using `step_env(action)`  
- Verifies all claims made by the White Agent

### **Judging & Scoring**
The Green Agent computes:
- Task completion score  
- Reward score  
- Efficiency / robustness  
- Policy compliance  
- Logging quality  
- Stability (crash detection)  

It also produces:
- A structured event log  
- A JSON final report with all metrics  

The Green Agent never searches or shops itself—  
it only **verifies**, **steps**, **logs**, and **scores**.

---

## 2.3 How the WebShop Environment Connects to AgentBeats

AgentBeats does not ship a WebShop backend. In this submission the **Green Agent hosts the WebShop text environment internally** (Gym-style), so no separate WebShop server is required for the benchmark to run.

### How it works

- When a battle starts, the **Green Agent** calls `setup_env()` and loads the JSON catalog.
- The **White Agent tool server** exposes only canonical WebShop actions (`search[…]`, `click[…]`, `select`, `buy`) to the LLM.
- Each White action is forwarded to Green, and Green executes it via `step_env(action)`.
- Green verifies legality (arg in clickables, Buy Now only on product page), logs the transition, and updates scores.

### Optional: external HTML WebShop server (only for UI/debug)

If you want to browse the catalog in a browser, you *may* start a standalone WebShop HTML server from the original WebShop repo. This is **not used for scoring**; it is purely for visualization.

If you do this, set:
```bash
export WEBSHOP_PORT=3000
```
so any helper scripts can link to:
`http://127.0.0.1:$WEBSHOP_PORT`.

### Summary

1. **Start AgentBeats backend**
2. **Load scenario (agents auto‑start)**
3. **Run scenario**

That’s enough for a full benchmark run.

---

# 3. Environment Setup

Activate your venv:

```bash
cd <YOUR_AGENTBEATS_PATH>
source venv/bin/activate
export OPENAI_API_KEY="sk-proj-…"
```

Optional for local development:

```bash
export DISABLE_AUTH=1
```

---

# 4. Start the AgentBeats Backend

Run in its own terminal:

```bash
source venv/bin/activate
agentbeats run_backend \
  --host 127.0.0.1 \
  --backend_port 9000 \
  --mcp_port 9001 \
  --reload
```

Output should include:

```
Uvicorn running on http://127.0.0.1:9000
FastMCP running on port 9001
```

---

# 5. Load the WebShop Scenario (Automatic Agent Launch)

In a second terminal:

```bash
source venv/bin/activate
export OPENAI_API_KEY="sk-proj-…"

agentbeats load_scenario <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final \
  --launch-mode tmux \
  --register_agents \
  --backend http://127.0.0.1:9000
```

This automatically launches:
- Green Agent  
- WebShop-White Agent (from persona card defined in `scenario.toml`)  

All agents start inside a **tmux session**.

---

# 6. (Optional) Run the Frontend UI

If you want a graphical UI:

```bash
source venv/bin/activate
agentbeats run_frontend \
  --frontend_mode dev \
  --host 127.0.0.1 \
  --frontend_port 5173 \
  --backend_url http://127.0.0.1:9000
```

Then open:

```
http://127.0.0.1:5173
```

---

# 7. Validate Agents (Health Check)

```bash
curl -s http://127.0.0.1:9031/.well-known/agent-card.json | jq .
curl -s http://127.0.0.1:9051/.well-known/agent-card.json | jq .
```

The White Agent runs on the port specified in the selected white_agent_*.toml (default in this repo: 9051).

---

# 8. Run the Full WebShop Benchmark

After the backend is running:

```bash
source venv/bin/activate

agentbeats run_scenario \
  <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final \
  --backend  http://127.0.0.1:9000 \
  --frontend http://127.0.0.1:5173
```

This executes:
- White Agent → uses `white_agent_server.py` tools to emit canonical actions
- White tool server → forwards actions to Green
- Green Agent → steps the internal WebShop text env, verifies legality, logs + scores
- Scenario → aggregates and prints the final benchmark score

Note: The White Agent communicates directly with the Green Agent; there is no intermediary routing agent.

---

# 9. White-Agent Personas (Test Cases)

This scenario supports multiple **white-agent configurations**, such as:

```
white_agent_fast_success.toml
white_agent_exploratory.toml
white_agent_partial_failure.toml
white_agent_crash.toml
```

Each defines:
- Persona description  
- Instruction template  
- Execution style  

## 9.1 Switch Test Case via scenario.toml

### How Scenario-Based White-Agent Selection Works (Important)

AgentBeats does **not** allow selecting a white-agent persona at runtime.
Instead, the scenario itself specifies *which* white agent card should be used.

This means:

- You always run the scenario using the same command:
  ```bash
  agentbeats run_scenario <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final
  ```
- The **only difference** between test cases is this one line in `scenario.toml`:
  ```toml
  white_agent_card = "Test_cases_card/white_agent_X.toml"
  ```
- Changing this line changes which persona participates in the WebShop benchmark.
- The Green Agent and scenario logic remain exactly the same.

In other words:
> To test a new persona, you **edit one line** in `scenario.toml`, then run the same `run_scenario` command again.

This is the intended mechanism for persona-based test cases in AgentBeats.

Edit:

```toml
white_agent_card = "Test_cases_card/white_agent_fast_success.toml"
```

Swap it with e.g.:

```toml
white_agent_card = "Test_cases_card/white_agent_exploratory.toml"
```

Run again:

```bash
agentbeats run_scenario <YOUR_AGENTBEATS_PATH>/scenarios/webshop_benchmark_final
```

## 9.2 Debug a White Agent Alone (No Full Scenario)

```bash
agentbeats run Test_cases_card/white_agent_fast_success.toml \
  --launcher_host 127.0.0.1 \
  --launcher_port 9101 \
  --agent_host 127.0.0.1 \
  --agent_port 9102 \
  --model_type openai \
  --model_name o4-mini
```

Useful for developing `white_agent_tools.py`.

---

# 10. Using tmux

AgentBeats launches all agents in a tmux session:

### Attach:
```bash
tmux attach -t agentbeats-webshop-benchmark
```

### Navigation:
- `Ctrl+b n` → next window  
- `Ctrl+b p` → previous window  
- `Ctrl+b d` → detach  

---

# 11. Troubleshooting

| Issue | Fix |
|------|------|
| Backend unresponsive | `curl http://127.0.0.1:9000/health` |
| Ports stuck | `lsof -ti :9030-9051 | xargs kill -9` |
| Reset scenario | `tmux kill-session -t agentbeats-webshop-benchmark` |
| Agent not registered | Check `load_scenario` logs |

---

# 12. Submission Notes (for Evaluators)

This repository contains:
- All agent cards  
- All agent implementations  
- All persona test cases  
- A scenario.toml orchestrator  
- A standalone README  
- No external dependencies or SDK code  

The repository is **fully self-contained** and can be executed via:

```
agentbeats load_scenario scenario/
agentbeats run_scenario scenario/
```

with an external WebShop backend running at `http://localhost:3000`.

---

If you have questions or want an auto‑generated architecture diagram, feel free to ask!
