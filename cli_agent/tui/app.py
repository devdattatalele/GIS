"""GIS split-pane TUI application."""

import os

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer

from cli_agent.tui.widgets import GISHeader, ActivityLog, DiffViewer
from cli_agent.tui.bridge import DisplayBridge


class GISApp(App):
    """Split-pane TUI for the GitHub Issue Solver agent."""

    TITLE = "gis"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "quit"),
        ("v", "toggle_verbose", "verbose"),
        ("tab", "focus_next", "switch pane"),
    ]

    def __init__(self, issue_url: str, config, services: dict,
                 workdir: str, max_iterations: int = 30,
                 auto_pr: bool = False, verbose: bool = False, **kw):
        super().__init__(**kw)
        self.issue_url = issue_url
        self.config = config
        self.services = services
        self.workdir = workdir
        self.max_iterations = max_iterations
        self.auto_pr = auto_pr
        self.verbose = verbose
        self._bridge: DisplayBridge | None = None

    def compose(self) -> ComposeResult:
        yield GISHeader(id="header")
        with Horizontal(id="main-content"):
            yield ActivityLog(id="activity-log")
            yield DiffViewer(id="diff-viewer")
        yield Footer()

    def on_mount(self) -> None:
        self._mute_all_loggers()
        self._start_agent()

    def _mute_all_loggers(self) -> None:
        """Kill all noisy loggers — only file logging survives."""
        import logging
        import warnings
        warnings.filterwarnings("ignore")
        os.environ["GRPC_VERBOSITY"] = "NONE"
        os.environ["ABSL_MIN_LOG_LEVEL"] = "99"
        for name in ["absl", "grpc", "google", "urllib3", "httpx",
                      "httpcore", "chromadb", "langchain", "langsmith",
                      "opentelemetry", "fastembed", "onnxruntime"]:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        try:
            from loguru import logger
            logger.remove()
            logger.add(os.path.join(self.workdir, "gis.log"),
                       rotation="5 MB", level="DEBUG")
        except ImportError:
            pass

    @work(thread=True, exclusive=True)
    def _start_agent(self) -> None:
        from cli_agent.agent import AgentRunner

        bridge = DisplayBridge(app=self, verbose=self.verbose)
        self._bridge = bridge

        runner = AgentRunner(
            config=self.config,
            services=self.services,
            workdir=self.workdir,
            max_iterations=self.max_iterations,
            auto_pr=self.auto_pr,
            verbose=self.verbose,
            display=bridge,
        )

        try:
            result = runner.run(self.issue_url)
            self.call_from_thread(self._on_done, result)
        except Exception as e:
            self.call_from_thread(self._on_error, str(e))

    def _on_done(self, result) -> None:
        h = self.query_one("#header", GISHeader)
        h.status_text = "done" if result.success else "failed"
        h.is_spinning = False
        self.query_one("#activity-log", ActivityLog).write_final(result)

    def _on_error(self, msg: str) -> None:
        h = self.query_one("#header", GISHeader)
        h.status_text = "error"
        h.is_spinning = False
        self.query_one("#activity-log", ActivityLog).write_error(msg)

    def action_toggle_verbose(self) -> None:
        self.verbose = not self.verbose
        if self._bridge:
            self._bridge.verbose = self.verbose
        tag = "on" if self.verbose else "off"
        self.query_one("#activity-log", ActivityLog).write(
            f"  verbose {tag}", shrink=False
        )
