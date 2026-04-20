"""Left pane: scrolling activity log."""

from textual.widgets import RichLog
from rich.text import Text

# Tool → color (Catppuccin Mocha palette)
_C = {
    "bash": "#f9e2af",
    "read_file": "#89b4fa",
    "write_file": "#cba6f7",
    "edit_file": "#cba6f7",
    "analyze_issue": "#74c7ec",
    "generate_patches": "#74c7ec",
    "search_codebase": "#a6e3a1",
    "search_learnings": "#a6e3a1",
    "ingest_repo": "#74c7ec",
    "get_repo_status": "#585b70",
    "show_diff": "#f9e2af",
}


class ActivityLog(RichLog):
    """Streaming log of agent reasoning, tool calls, and results."""

    def on_mount(self) -> None:
        self.write(Text.from_markup(
            "[bold #4ec9b0]gis[/bold #4ec9b0] [#a6adc8]agent started[/#a6adc8]"
        ))

    def write_step(self, n: int, total: int) -> None:
        self.write(Text.from_markup(
            f"\n[#585b70]{'─' * 36}  step {n}/{total}[/#585b70]"
        ))
        self.scroll_end(animate=False)

    def write_reasoning(self, text: str) -> None:
        self.write(Text.from_markup("[#585b70]thinking ↓[/#585b70]"))
        for line in text.strip().splitlines()[:12]:
            self.write(Text(f"  {line}", style="#a6adc8"))
        self.scroll_end(animate=False)

    def write_tool_call(self, name: str, args: dict) -> None:
        c = _C.get(name, "#cdd6f4")
        summary = _tool_summary(name, args)
        self.write(Text.from_markup(f"  [{c}]▸ {name}[/{c}]  [#585b70]{summary}[/#585b70]"))
        self.scroll_end(animate=False)

    def write_tool_result(self, name: str, short: str) -> None:
        c = _C.get(name, "#cdd6f4")
        line = short[:100].replace("\n", " ")
        if len(short) > 100:
            line += "…"
        self.write(Text.from_markup(f"    [{c}]←[/{c}] [#585b70]{line}[/#585b70]"))
        self.scroll_end(animate=False)

    def write_final(self, result) -> None:
        ok = result.success
        icon = "[#a6e3a1]✓[/#a6e3a1]" if ok else "[#f38ba8]✗[/#f38ba8]"
        label = "[#a6e3a1]success[/#a6e3a1]" if ok else "[#f38ba8]failed[/#f38ba8]"
        self.write(Text.from_markup(f"\n[#585b70]{'━' * 36}[/#585b70]"))
        self.write(Text.from_markup(
            f"  {icon} {label}  "
            f"[#585b70]steps[/#585b70] {result.iterations}  "
            f"[#585b70]tools[/#585b70] {result.tool_count}  "
            f"[#585b70]time[/#585b70] {result.duration:.0f}s"
        ))
        if result.pr_url:
            self.write(Text.from_markup(f"  [#89b4fa]pr[/#89b4fa] {result.pr_url}"))
        if result.summary:
            self.write(Text(f"  {result.summary[:160]}", style="#a6adc8"))
        self.scroll_end(animate=False)

    def write_error(self, msg: str) -> None:
        self.write(Text.from_markup(f"  [#f38ba8]error:[/#f38ba8] {msg}"))
        self.scroll_end(animate=False)


def _tool_summary(name: str, args: dict) -> str:
    if name == "bash" and "command" in args:
        return f"$ {args['command'][:80]}"
    if name in ("read_file", "write_file", "edit_file") and "path" in args:
        return args["path"]
    if name in ("analyze_issue", "ingest_repo"):
        return args.get("url", args.get("repo_name", ""))
    parts = [f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:3]]
    return ", ".join(parts)
