"""GIS CLI — entry point.

    gis              interactive menu
    gis setup        configure API keys
    gis status       show config
    gis <url>        resolve a GitHub issue
"""

import os
import sys
from pathlib import Path

import click

from cli_agent import __version__
from cli_agent.config import ensure_gh_cli_available


def _silence_everything():
    """Kill ALL noisy loggers before any service imports."""
    import logging
    import warnings

    # Suppress all warnings (grpc, absl, etc.)
    warnings.filterwarnings("ignore")

    # Suppress absl/grpc noise
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "99"

    # Kill standard logging noise
    for name in ["absl", "grpc", "google", "urllib3", "httpx",
                 "httpcore", "chromadb", "langchain", "langsmith",
                 "opentelemetry", "fastembed", "onnxruntime"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False

    # Kill loguru → redirect to file only
    try:
        from loguru import logger
        logger.remove()  # remove ALL handlers (kills stderr output)
    except ImportError:
        pass


class _CLI(click.Group):
    """Routes bare URLs to the run subcommand."""

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            if args[0].startswith("http"):
                args = ["run"] + args
        return super().parse_args(ctx, args)


@click.group(cls=_CLI, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="gis")
@click.pass_context
def main(ctx):
    """gis — autonomous GitHub issue solver."""
    if ctx.invoked_subcommand is None:
        _interactive()


# ── Interactive menu ────────────────────────────────────────────

def _interactive():
    from rich.console import Console
    from cli_agent.prompts_tui import select, text_input

    c = Console()
    c.print()
    c.print(f"  [bold #4ec9b0]gis[/bold #4ec9b0]  [#a6adc8]v{__version__}[/#a6adc8]\n")

    idx = select(
        "what would you like to do?",
        ["Resolve a GitHub issue", "Setup API keys", "Show configuration"],
        ["paste an issue URL", "providers, tokens, settings", "check current config"],
    )

    if idx == 1:
        from cli_agent.setup import run_setup
        run_setup()
        return
    if idx == 2:
        from cli_agent.setup import show_status
        show_status()
        return

    url = text_input("paste the issue URL")
    if not url or not url.startswith("http"):
        c.print("  [#f38ba8]invalid URL[/#f38ba8]")
        return

    mode = select("display mode", ["TUI (split pane)", "Classic (scrolling)"],
                  ["structured layout", "plain Rich output"])

    _launch(issue_url=url, tui=(mode == 0))


# ── Shared launch logic ────────────────────────────────────────

def _init_services(env_file, model, provider):
    """Initialize services with all noise suppressed and a spinner."""
    from rich.console import Console
    from rich.spinner import Spinner
    from rich.live import Live

    _silence_everything()

    console = Console()
    spinner = Spinner("dots", text="[#585b70]  loading services…[/#585b70]")

    with Live(spinner, console=console, refresh_per_second=10):
        from cli_agent.services import initialize_services
        config, services = initialize_services(
            env_file=env_file, model=model, provider=provider,
        )

    # Re-silence loguru after services may have re-added handlers
    try:
        from loguru import logger
        logger.remove()
    except ImportError:
        pass

    return config, services


def _launch(*, issue_url: str, env_file: str | None = None,
            model: str | None = None, provider: str | None = None,
            max_iterations: int = 30, workdir: str | None = None,
            auto_pr: bool = False, verbose: bool = False, tui: bool = True):

    if workdir is None:
        workdir = str(Path.cwd() / "gis_workspace")
    workdir = str(Path(workdir).resolve())
    os.makedirs(workdir, exist_ok=True)

    if not ensure_gh_cli_available():
        click.echo("warning: gh CLI not found — PR creation will fail", err=True)

    if env_file is None:
        from cli_agent.setup import resolve_env_file
        env_file = resolve_env_file()

    if env_file is None:
        click.echo("no config found. run: gis setup", err=True)
        sys.exit(1)

    try:
        config, services = _init_services(env_file, model, provider)
    except Exception as e:
        click.echo(f"\n  [init failed] {e}", err=True)
        click.echo("  run: gis setup", err=True)
        sys.exit(1)

    if tui:
        from cli_agent.tui import GISApp
        GISApp(issue_url=issue_url, config=config, services=services,
               workdir=workdir, max_iterations=max_iterations,
               auto_pr=auto_pr, verbose=verbose).run()
    else:
        from cli_agent.agent import AgentRunner
        runner = AgentRunner(config=config, services=services, workdir=workdir,
                             max_iterations=max_iterations, auto_pr=auto_pr,
                             verbose=verbose)
        try:
            result = runner.run(issue_url)
        except KeyboardInterrupt:
            click.echo("\naborted.")
            sys.exit(0)
        runner.display.show_final_result(result)
        sys.exit(0 if result.success else 1)


# ── Subcommands ─────────────────────────────────────────────────

@main.command()
def setup():
    """Configure API keys and settings."""
    from cli_agent.setup import run_setup
    run_setup()


@main.command()
def status():
    """Show current configuration."""
    from cli_agent.setup import show_status
    show_status()


@main.command()
@click.argument("issue_url")
@click.option("--env-file", default=None, help="Path to .env file")
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["gemini", "claude", "grok", "openai", "ollama"], case_sensitive=False), default=None)
@click.option("--max-iterations", default=30, help="Max agent iterations")
@click.option("--workdir", default=None, help="Working directory")
@click.option("--auto-pr", is_flag=True, help="Skip PR confirmation")
@click.option("-v", "--verbose", is_flag=True, help="Full tool output")
@click.option("--tui/--no-tui", default=True, help="TUI mode (default: on)")
def run(issue_url, **kw):
    """Resolve a GitHub issue. Example: gis run https://github.com/o/r/issues/1"""
    _launch(issue_url=issue_url, **kw)


@main.command()
@click.option("--repo", default=None, help="Filter to specific repo (e.g. owner/repo)")
@click.option("--embedding", default=None, help="Override embedding provider (fastembed/google)")
@click.option("--dataset", default=None, help="Path to golden dataset JSON")
@click.option("--output", default=None, help="Output report JSON path")
@click.option("--top-k", type=int, default=5, help="Chunks to retrieve per query")
def eval(repo, embedding, dataset, output, top_k):
    """Run RAG evaluation against ingested repos."""
    _silence_everything()
    from evals.run_eval import run_evaluation
    run_evaluation(
        repo_filter=repo,
        embedding_provider=embedding,
        dataset_path=dataset,
        output_path=output,
        k=top_k,
    )


@main.command(name="eval-report")
@click.option("--input", "input_path", default=None, help="Path to eval report.json")
@click.option("--output", "output_path", default=None, help="Output PDF path")
def eval_report(input_path, output_path):
    """Generate PDF report from RAG evaluation results."""
    from evals.generate_report import generate_pdf_report
    pdf_path = generate_pdf_report(input_path=input_path, output_path=output_path)
    click.echo(f"  PDF saved: {pdf_path}")


if __name__ == "__main__":
    main()
