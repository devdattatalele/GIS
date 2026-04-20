"""Interactive setup wizard and status display."""

import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cli_agent.prompts_tui import select, text_input, _mask, _ACCENT, _DIM, _OK

console = Console()

CONFIG_DIR = Path.home() / ".config" / "gis"
CONFIG_FILE = CONFIG_DIR / "config.env"

# ── Provider definitions ──────────────────────────────────────
_PROVIDERS = [
    ("gemini",  "Gemini (Google)",     "free tier, fast",           "GOOGLE_API_KEY",    "https://makersuite.google.com/app/apikey"),
    ("claude",  "Claude (Anthropic)",  "excellent code quality",    "ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
    ("grok",    "Grok (xAI)",          "strong reasoning",          "XAI_API_KEY",       "https://console.x.ai/"),
    ("openai",  "OpenAI",              "gpt-4o",                    "OPENAI_API_KEY",    "https://platform.openai.com/api-keys"),
    ("ollama",  "Ollama (local)",      "offline, no API key",       None,                None),
]

_PROVIDER_IDS   = [p[0] for p in _PROVIDERS]
_PROVIDER_NAMES = [p[1] for p in _PROVIDERS]
_PROVIDER_DESCS = [p[2] for p in _PROVIDERS]


def resolve_env_file(explicit: str | None = None) -> str | None:
    """Resolve config file: explicit -> ~/.config/gis -> project .env."""
    if explicit:
        return explicit
    if CONFIG_FILE.exists():
        return str(CONFIG_FILE)
    from cli_agent.config import get_project_root
    p = get_project_root() / ".env"
    return str(p) if p.exists() else None


def _load_existing() -> dict:
    vals = {}
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip()
    return vals


def run_setup():
    """Interactive setup wizard."""
    console.print()
    console.print(f"  [bold {_ACCENT}]gis[/bold {_ACCENT}] [bold]setup[/bold]\n")

    existing = _load_existing()
    cfg = {}

    # ── 1. LLM Provider ───────────────────────────────────────
    console.print(f"  [bold]1 · LLM Provider[/bold]\n")

    cur_provider = existing.get("LLM_PROVIDER", "gemini")
    default_idx = _PROVIDER_IDS.index(cur_provider) if cur_provider in _PROVIDER_IDS else 0

    idx = select("choose provider", _PROVIDER_NAMES, _PROVIDER_DESCS, default=default_idx)
    provider_id, _, _, env_key, key_url = _PROVIDERS[idx]
    cfg["LLM_PROVIDER"] = provider_id

    # ── 2. API Key ─────────────────────────────────────────────
    console.print(f"  [bold]2 · API Key[/bold]\n")

    if env_key:
        if key_url:
            console.print(f"  [{_DIM}]{key_url}[/{_DIM}]\n")
        key = text_input(f"{env_key}", default=existing.get(env_key, ""), mask=True)
        if key:
            cfg[env_key] = key
    else:
        console.print(f"  [{_DIM}]no API key needed for {provider_id}[/{_DIM}]\n")

    # Preserve existing keys for other providers
    for _, _, _, ek, _ in _PROVIDERS:
        if ek and ek != env_key and existing.get(ek):
            cfg[ek] = existing[ek]

    # ── 3. GitHub Token ────────────────────────────────────────
    console.print(f"  [bold]3 · GitHub Token[/bold]\n")
    console.print(f"  [{_DIM}]scopes: repo, read:org · https://github.com/settings/tokens[/{_DIM}]\n")
    gh = text_input("GitHub Token", default=existing.get("GITHUB_TOKEN", ""), mask=True)
    if gh:
        cfg["GITHUB_TOKEN"] = gh

    # ── 4. Embeddings ──────────────────────────────────────────
    console.print(f"  [bold]4 · Embeddings[/bold]\n")
    cur_e = 0 if existing.get("EMBEDDING_PROVIDER", "fastembed") == "fastembed" else 1
    ei = select("embedding model", ["FastEmbed (offline)", "Google Embeddings"],
                ["no API cost", "uses Gemini quota"], default=cur_e)
    cfg["EMBEDDING_PROVIDER"] = "fastembed" if ei == 0 else "google"

    # ── Defaults ───────────────────────────────────────────────
    for k, d in [("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
                 ("CHROMA_PERSIST_DIR", "./chroma_db"), ("ENABLE_PATCH_GENERATION", "true"),
                 ("LOG_LEVEL", "INFO"), ("MAX_ISSUES", "100"), ("MAX_PRS", "15"), ("MAX_FILES", "50")]:
        cfg.setdefault(k, existing.get(k, d))

    # ── Save ───────────────────────────────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# gis config", ""]
    for k, v in cfg.items():
        lines.append(f"{k}={v}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n")

    console.print(f"  [{_OK}]saved[/{_OK}] {CONFIG_FILE}\n")

    # ── Verify ─────────────────────────────────────────────────
    gh_ok = shutil.which("gh") is not None
    checks = []

    if env_key:
        k = cfg.get(env_key, "")
        checks.append(("[green]ok[/green]" if k and len(k) > 10 else "[red]missing[/red]", env_key.lower().replace("_", " ")))

    t = cfg.get("GITHUB_TOKEN", "")
    checks.append(("[green]ok[/green]" if t else "[red]missing[/red]", "github token"))
    checks.append(("[green]ok[/green]" if gh_ok else "[yellow]optional[/yellow]", "github cli (gh)"))

    for icon, label in checks:
        console.print(f"  {icon} {label}")

    console.print(f"\n  [{_DIM}]run [bold]gis[/bold] to start[/{_DIM}]\n")


def show_status():
    """Print current configuration status."""
    console.print()
    console.print(f"  [bold {_ACCENT}]gis[/bold {_ACCENT}] [bold]status[/bold]\n")

    env = resolve_env_file()
    src = str(env) if env else "none"

    cfg = {}
    if env and Path(env).exists():
        for line in Path(env).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()

    for key in ["LLM_PROVIDER", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
                "XAI_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN",
                "EMBEDDING_PROVIDER"]:
        ev = os.environ.get(key)
        if ev and key not in cfg:
            cfg[key] = ev

    t = Table(show_header=True, border_style="#313244", header_style="bold #a6adc8",
              row_styles=["", "#1e1e2e"])
    t.add_column("setting", style="bold")
    t.add_column("value")
    t.add_column("source", style="#585b70")

    t.add_row("config", src, "")
    t.add_row("llm", cfg.get("LLM_PROVIDER", "-"), src)
    t.add_row("google key", _mask(cfg.get("GOOGLE_API_KEY", "")) or "-", src)
    t.add_row("anthropic key", _mask(cfg.get("ANTHROPIC_API_KEY", "")) or "-", src)
    t.add_row("xai key", _mask(cfg.get("XAI_API_KEY", "")) or "-", src)
    t.add_row("openai key", _mask(cfg.get("OPENAI_API_KEY", "")) or "-", src)
    t.add_row("github token", _mask(cfg.get("GITHUB_TOKEN", "")) or "-", src)
    t.add_row("embeddings", cfg.get("EMBEDDING_PROVIDER", "-"), src)
    gh = shutil.which("gh") is not None
    t.add_row("gh cli", "[green]ok[/green]" if gh else "[yellow]missing[/yellow]", "PATH")

    console.print(t)
    console.print()

    if not env:
        console.print(f"  [{_DIM}]run [bold]gis setup[/bold] to configure[/{_DIM}]\n")
