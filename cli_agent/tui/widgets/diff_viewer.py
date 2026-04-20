"""Right pane: syntax-highlighted diff / file viewer."""

import os
from textual.containers import Vertical
from textual.widgets import Static, RichLog
from rich.syntax import Syntax
from rich.text import Text

_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".html": "html", ".css": "css",
    ".sh": "bash", ".rs": "rust", ".go": "go",
    ".java": "java", ".rb": "ruby", ".sql": "sql",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
}


class DiffViewer(Vertical):
    """Syntax-highlighted content viewer for diffs and file contents."""

    def compose(self):
        yield Static("waiting for content…", id="diff-title")
        yield RichLog(id="diff-content", highlight=True, markup=True)

    def set_content(self, content: str, title: str = "", is_diff: bool = False) -> None:
        title_w = self.query_one("#diff-title", Static)
        body = self.query_one("#diff-content", RichLog)

        title_w.update(title or ("diff" if is_diff else "file"))
        body.clear()

        if not content or not content.strip():
            body.write(Text("(empty)", style="#585b70"))
            return

        if is_diff or content.lstrip().startswith("diff --git"):
            try:
                body.write(Syntax(content, "diff", theme="monokai",
                                  line_numbers=True, word_wrap=False))
            except Exception:
                body.write(Text(content))
        else:
            lang = _EXT.get(os.path.splitext(title)[1].lower(), "text") if title else "text"
            try:
                body.write(Syntax(content, lang, theme="monokai",
                                  line_numbers=True, word_wrap=False))
            except Exception:
                body.write(Text(content))
