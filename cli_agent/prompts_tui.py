"""Arrow-key interactive prompts for terminal.

Uses raw tty mode (stdlib) + Rich for styled rendering.
No extra dependencies.
"""

import sys
import tty
import termios

from rich.console import Console
from rich.text import Text

console = Console()

# Catppuccin Mocha accents
_ACCENT = "#4ec9b0"
_DIM = "#585b70"
_TEXT = "#cdd6f4"
_OK = "#a6e3a1"


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3, "esc")
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select(prompt: str, options: list[str],
           descriptions: list[str] | None = None, default: int = 0) -> int:
    """Arrow-key selector. Returns chosen index."""
    sel = default
    descs = descriptions or [""] * len(options)
    n = len(options)

    console.print(f"  [{_DIM}]{prompt}[/{_DIM}]\n")

    def _draw():
        for i, (opt, desc) in enumerate(zip(options, descs)):
            if i == sel:
                console.print(f"    [{_ACCENT}]▸ {opt}[/{_ACCENT}]  [{_DIM}]{desc}[/{_DIM}]")
            else:
                console.print(f"    [{_DIM}]  {opt}  {desc}[/{_DIM}]")

    _draw()

    while True:
        key = _read_key()
        if key == "up":
            sel = (sel - 1) % n
        elif key == "down":
            sel = (sel + 1) % n
        elif key == "enter":
            sys.stdout.write(f"\033[{n}A\033[J")
            sys.stdout.flush()
            console.print(f"    [{_OK}]▸ {options[sel]}[/{_OK}]\n")
            return sel
        elif key == "esc":
            sys.stdout.write(f"\033[{n}A\033[J")
            sys.stdout.flush()
            console.print(f"    [{_OK}]▸ {options[default]}[/{_OK}]\n")
            return default
        else:
            continue

        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()
        _draw()


def text_input(prompt: str, default: str = "", mask: bool = False) -> str:
    """Styled text input. Returns entered value or default."""
    hint = ""
    if default and mask:
        hint = f" [{_DIM}]({_mask(default)})[/{_DIM}]"
    elif default:
        hint = f" [{_DIM}](enter to keep current)[/{_DIM}]"

    console.print(f"  [{_DIM}]{prompt}[/{_DIM}]{hint}")
    try:
        val = console.input(f"  [{_ACCENT}]→[/{_ACCENT}] ").strip()
    except (KeyboardInterrupt, EOFError):
        val = ""

    val = val or default
    display = _mask(val) if mask and val else (val or f"[{_DIM}]empty[/{_DIM}]")
    console.print(f"  [{_OK}]✓[/{_OK}] {display}\n")
    return val


def _mask(s: str) -> str:
    if not s or len(s) < 12:
        return "••••"
    return s[:4] + "•••" + s[-4:]
