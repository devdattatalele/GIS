"""Compact single-line header bar."""

from textual.reactive import reactive
from textual.widget import Widget
from rich.text import Text

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class GISHeader(Widget):
    """Single-line header: logo · url · step · status."""

    DEFAULT_CSS = """
    GISHeader {
        height: 1;
        background: #181825;
        color: #cdd6f4;
    }
    """

    issue_url: reactive[str] = reactive("")
    step_text: reactive[str] = reactive("")
    status_text: reactive[str] = reactive("ready")
    is_spinning: reactive[bool] = reactive(False)

    _frame: int = 0
    _timer = None

    def watch_is_spinning(self, value: bool) -> None:
        if value and self._timer is None:
            self._timer = self.set_interval(1 / 12, self._tick)
        elif not value and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._frame = 0

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_FRAMES)
        self.refresh()

    def render(self) -> Text:
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(" gis ", style="bold #1e1e2e on #4ec9b0")

        if self.issue_url:
            t.append(f"  {self.issue_url} ", style="#89b4fa underline")

        if self.step_text:
            t.append(f" {self.step_text} ", style="#a6adc8")

        t.append(" ")
        if self.is_spinning:
            t.append(f"{_FRAMES[self._frame]} ", style="#f9e2af")
            t.append(self.status_text, style="#f9e2af")
        else:
            t.append(f"● {self.status_text}", style="#a6e3a1")

        return t
