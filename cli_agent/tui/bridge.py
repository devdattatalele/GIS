"""Thread-safe bridge: agent thread → TUI widgets via call_from_thread."""

import threading
from contextlib import contextmanager

from cli_agent.display import MAX_DISPLAY_LEN

_VIEWER_TOOLS = {"show_diff", "read_file", "analyze_issue", "edit_file"}


class DisplayBridge:
    """Drop-in Display replacement that routes to Textual widgets."""

    def __init__(self, app, verbose: bool = False):
        self.app = app
        self.verbose = verbose

    def _log(self):
        return self.app.query_one("#activity-log")

    def _viewer(self):
        return self.app.query_one("#diff-viewer")

    def _header(self):
        return self.app.query_one("#header")

    def _truncate(self, text: str, limit: int = MAX_DISPLAY_LEN) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n… ({len(text) - limit} chars hidden)"

    # ── Display API ──────────────────────────────────────────

    def show_banner(self) -> None:
        pass  # header is always visible — no separate banner needed

    def show_agent_start(self, url: str, workdir: str) -> None:
        def _do():
            self._header().issue_url = url
            self._header().status_text = "initializing"
        self.app.call_from_thread(_do)

    def show_iteration_header(self, n: int, max_iter: int) -> None:
        def _do():
            self._header().step_text = f"{n}/{max_iter}"
            self._log().write_step(n, max_iter)
        self.app.call_from_thread(_do)

    @contextmanager
    def show_thinking_spinner(self):
        def _on():
            h = self._header()
            h.is_spinning = True
            h.status_text = "thinking"

        def _off():
            h = self._header()
            h.is_spinning = False
            h.status_text = "running"

        self.app.call_from_thread(_on)
        try:
            yield
        finally:
            self.app.call_from_thread(_off)

    def show_agent_reasoning(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.app.call_from_thread(lambda: self._log().write_reasoning(text))

    def show_tool_call(self, name: str, args: dict) -> None:
        def _do():
            self._header().status_text = name
            self._log().write_tool_call(name, args)
        self.app.call_from_thread(_do)

    def show_tool_result(self, name: str, result: str) -> None:
        short = result if self.verbose else self._truncate(result)
        full = result

        def _do():
            self._log().write_tool_result(name, short)
            if name in _VIEWER_TOOLS:
                is_diff = name == "show_diff" or full.lstrip().startswith("diff --git")
                title = name
                if name == "read_file":
                    first = full.split("\n")[0] if full else ""
                    title = first if len(first) < 100 else name
                self._viewer().set_content(full, title=title, is_diff=is_diff)
        self.app.call_from_thread(_do)

    def show_agent_response(self, text: str) -> None:
        def _do():
            self._log().write_reasoning(text)
            self._header().status_text = "done"
            self._header().is_spinning = False
        self.app.call_from_thread(_do)

    def show_final_result(self, result) -> None:
        def _do():
            self._log().write_final(result)
            self._header().status_text = "done" if result.success else "failed"
            self._header().is_spinning = False
        self.app.call_from_thread(_do)

    def show_error(self, message: str) -> None:
        self.app.call_from_thread(lambda: self._log().write_error(message))

    def confirm_pr_creation(self) -> bool:
        event = threading.Event()
        holder = [False]

        def _cb(confirmed: bool) -> None:
            holder[0] = confirmed
            event.set()

        def _push():
            from cli_agent.tui.widgets.confirm_modal import ConfirmModal
            self.app.push_screen(ConfirmModal(), callback=_cb)

        self.app.call_from_thread(_push)
        event.wait()
        return holder[0]
