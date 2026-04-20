# CLAUDE.md

## Project: GIS — GitHub Issue Solver

Autonomous CLI agent that resolves GitHub issues end-to-end: ingests repo into RAG knowledge base, analyzes the issue, generates patches, and creates PRs.

## Commands

```bash
gis                     # Interactive menu
gis setup               # Configure API keys and providers
gis status              # Show current configuration
gis <url>               # Resolve a GitHub issue (TUI mode)
gis <url> --no-tui      # Resolve with classic Rich output
gis run <url> --provider grok --model grok-3  # Override provider
pip install -e .        # Install from source
python -m evals.run_eval  # Run RAG evaluation
```

## Architecture

```
cli_agent/              # CLI package (entry: main.py)
├── main.py             # Click CLI — interactive menu + subcommands
├── agent.py            # ReAct agent loop (sync, drives LLM + tools)
├── tools.py            # 11 LangChain tools (bash, read/write/edit, RAG)
├── display.py          # Rich console output (--no-tui fallback)
├── services.py         # Service initialization bridge
├── prompts.py          # Agent system prompt
├── prompts_tui.py      # Arrow-key selector (stdlib tty/termios)
├── setup.py            # Setup wizard (5 providers)
└── tui/                # Textual split-pane TUI
    ├── app.py          # GISApp — main Textual application
    ├── bridge.py       # Thread-safe Display adapter (agent→TUI)
    └── widgets/        # header, activity_log, diff_viewer, confirm_modal

src/github_issue_solver/ # Core service layer
├── config.py            # Config with env vars, provider detection
├── server.py            # MCP server (FastMCP, 17 tools)
├── services/
│   ├── llm_service.py   # LiteLLM router (gemini/claude/grok/openai/ollama)
│   ├── embedding_service.py  # FastEmbed (offline) / Google embeddings
│   ├── ingestion_service.py  # 4-step repo ingestion
│   ├── analysis_service.py   # RAG-powered issue analysis
│   ├── patch_service.py      # AI patch generation
│   ├── learning_service.py   # Per-repo learnings & never-do rules
│   └── health_service.py     # System health monitoring

evals/                   # RAG evaluation framework
├── run_eval.py          # LLM-as-judge evaluation runner
└── golden_dataset.json  # Ground-truth Q&A pairs
```

## Key Patterns

- **LiteLLM routing**: Single `PROVIDERS` registry maps provider → (prefix, env_var, default_model). ChatLiteLLM wraps all providers in LangChain-compatible BaseChatModel.
- **Threading bridge**: Agent loop is sync. TUI is async. Bridge uses `app.call_from_thread()`.
- **Display interface**: `Display` (Rich) and `DisplayBridge` (TUI) share same API.
- **Config resolution**: `--env-file` → `~/.config/gis/config.env` → project `.env`.
- **Collection naming**: User-isolated (`user_X_repo_code`) in v4+; legacy (`repo_code`) in v3. Eval runner handles both.

## Development

- Entry point: `pyproject.toml` → `gis = "cli_agent.main:main"`
- Python >= 3.10
- Key deps: litellm, langchain, chromadb, textual, rich, click, fastembed
