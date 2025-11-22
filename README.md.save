---
## 2.3 How the WebShop Server Connects to AgentBeats

AgentBeats does not include the WebShop backend internally. Instead, the Blue Agent or White-Agent tools connect directly to the external WebShop server you start from the standalone WebShop repository.

### How it works

- The external WebShop repository exposes a REST API at  
  `http://localhost:<WEBSHOP_PORT>/api/...`
- Your **white agent tools** (in `white_agent_tools.py`) call this API directly.
- Your **blue agent card** (if used) is configured with the same base URL.
- The **Green Agent** evaluates the actions taken against this external server.

### Required environment variable

Set the port where the external server runs:

```bash
export WEBSHOP_PORT=3000
```

AgentBeats will use it to construct full WebShop URLs:

```
http://127.0.0.1:$WEBSHOP_PORT/api/query
http://127.0.0.1:$WEBSHOP_PORT/api/open_item
```

### Summary

1. **Start the WebShop server** from the WebShop repository  
2. **Export WEBSHOP_PORT**  
3. **Start AgentBeats backend**  
4. **Load and run the scenario**  
5. The agents will automatically connect to the running WebShop instance

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
- Blue Agent  
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
curl -s http://127.0.0.1:9041/.well-known/agent-card.json | jq .
curl -s http://127.0.0.1:9051/.well-known/agent-card.json | jq .
```

Each should return valid JSON.

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
- White Agent → calls its own tool server → forwarded to Green Agent → Green Agent steps the WebShop environment internally  
- Blue Agent → routes queries  
- Green Agent → evaluates trajectory  
- Scenario → aggregates final benchmark score  

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
