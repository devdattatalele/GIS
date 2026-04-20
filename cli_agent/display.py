"""Rich terminal display for the autonomous agent."""

import re
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()

# Tool name → color mapping
TOOL_COLORS = {
    "bash": "yellow",
    "read_file": "cyan",
    "write_file": "magenta",
    "edit_file": "magenta",
    "analyze_issue": "blue",
    "generate_patches": "blue",
    "search_codebase": "green",
    "search_learnings": "green",
    "ingest_repo": "blue",
    "get_repo_status": "dim",
    "show_diff": "yellow",
}

MAX_DISPLAY_LEN = 2000


class Display:
    """Rich terminal output for agent activity."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def show_banner(self):
        banner = Text()
        banner.append("GIS", style="bold cyan")
        banner.append(" - GitHub Issue Solver Agent", style="bold white")
        console.print(Panel(banner, border_style="cyan", padding=(1, 2)))

    def show_agent_start(self, url: str, workdir: str):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Issue", f"[bold]{url}[/bold]")
        table.add_row("Workdir", f"[dim]{workdir}[/dim]")
        console.print(Panel(table, title="[bold]Agent Starting[/bold]", border_style="blue"))

    def show_iteration_header(self, n: int, max_iter: int):
        console.print(f"\n[bold dim]--- Step {n}/{max_iter} ---[/bold dim]")

    def show_thinking_spinner(self) -> Live:
        """Return a Rich Live context manager with a spinner."""
        spinner = Spinner("dots", text="[dim]Thinking...[/dim]")
        return Live(spinner, console=console, refresh_per_second=10)

    def show_agent_reasoning(self, text: str):
        if not text or not text.strip():
            return
        console.print(Panel(
            Markdown(text),
            title="[dim]Reasoning[/dim]",
            border_style="dim",
            padding=(0, 1),
        ))

    def show_tool_call(self, name: str, args: dict):
        color = TOOL_COLORS.get(name, "white")
        args_display = ""
        if name == "bash" and "command" in args:
            args_display = f"$ {args['command']}"
        elif name in ("read_file", "write_file", "edit_file") and "path" in args:
            args_display = args["path"]
        elif name in ("analyze_issue", "ingest_repo", "get_repo_status"):
            args_display = args.get("url", args.get("repo_name", ""))
        else:
            parts = [f"{k}={repr(v)[:60]}" for k, v in args.items()]
            args_display = ", ".join(parts)

        console.print(f"  [{color}]> {name}[/{color}] [dim]{args_display}[/dim]")

    def show_tool_result(self, name: str, result: str):
        display_result = result if self.verbose else self._truncate(result)

        # Auto-detect diffs
        if name == "show_diff" or display_result.lstrip().startswith("diff --git"):
            try:
                console.print(Panel(
                    Syntax(display_result, "diff", theme="monokai"),
                    title=f"[dim]{name} result[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                ))
                return
            except Exception:
                pass

        console.print(Panel(
            Text(display_result),
            title=f"[dim]{name} result[/dim]",
            border_style="dim",
            padding=(0, 1),
        ))

    def show_agent_response(self, text: str):
        console.print(Panel(
            Markdown(text),
            title="[bold green]Agent Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    def show_final_result(self, result):
        table = Table(title="Summary", show_header=False, border_style="cyan")
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Status", "[green]Success[/green]" if result.success else "[red]Failed[/red]")
        table.add_row("Iterations", str(result.iterations))
        table.add_row("Tools Used", str(result.tool_count))
        table.add_row("Duration", f"{result.duration:.1f}s")
        if result.pr_url:
            table.add_row("PR URL", f"[link={result.pr_url}]{result.pr_url}[/link]")
        if result.summary:
            table.add_row("Summary", result.summary[:200])
        console.print(table)

    def show_error(self, message: str):
        console.print(f"[bold red]Error:[/bold red] {message}")

    def confirm_pr_creation(self) -> bool:
        """Ask user to confirm PR creation. Returns True if confirmed."""
        return console.input("\n[bold yellow]Create PR? [y/N]: [/bold yellow]").strip().lower() == "y"

    def _truncate(self, text: str, limit: int = MAX_DISPLAY_LEN) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"
