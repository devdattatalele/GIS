"""Modal for PR creation confirmation."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmModal(ModalScreen[bool]):
    """Yes/No modal — blocks the agent thread until answered."""

    BINDINGS = [("y", "yes", "Yes"), ("n", "no", "No"), ("escape", "no", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static("Create Pull Request?", id="confirm-title")
            yield Static("The agent wants to open a PR on GitHub.", id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes, create", variant="success", id="btn-yes")
                yield Button("Skip", variant="error", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
