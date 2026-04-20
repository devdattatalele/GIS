"""LangChain tool definitions for the autonomous agent.

Factory function returns 11 tools as closures over config/services/workdir.
"""

import asyncio
import json
import subprocess
from pathlib import Path

from langchain_core.tools import tool


MAX_BASH_OUTPUT = 15000
MAX_FILE_READ = 30000


def _run_async(coro):
    """Run an async coroutine from sync context safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — create a new thread to run it
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def create_tools(config, services: dict, workdir: str) -> list:
    """Create all agent tools as closures over shared state."""

    @tool
    def bash(command: str) -> str:
        """Run a shell command. Use for git, tests, gh CLI, and general shell operations.

        Args:
            command: The shell command to execute.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=workdir,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output:
                output = f"(exit code {result.returncode})"
            if len(output) > MAX_BASH_OUTPUT:
                output = output[:MAX_BASH_OUTPUT] + f"\n... ({len(output) - MAX_BASH_OUTPUT} chars truncated)"
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out after 120 seconds."
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def read_file(path: str) -> str:
        """Read a file's contents.

        Args:
            path: Absolute or relative (to workdir) file path.
        """
        try:
            file_path = Path(path) if Path(path).is_absolute() else Path(workdir) / path
            content = file_path.read_text(errors="replace")
            if len(content) > MAX_FILE_READ:
                content = content[:MAX_FILE_READ] + f"\n... ({len(content) - MAX_FILE_READ} chars truncated)"
            return content
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: Absolute or relative (to workdir) file path.
            content: The full file content to write.
        """
        try:
            file_path = Path(path) if Path(path).is_absolute() else Path(workdir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return f"OK: Wrote {len(content)} chars to {file_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of `old` with `new` in a file.

        Args:
            path: Absolute or relative (to workdir) file path.
            old: The exact text to find (first occurrence).
            new: The replacement text.
        """
        try:
            file_path = Path(path) if Path(path).is_absolute() else Path(workdir) / path
            content = file_path.read_text()
            if old not in content:
                return f"ERROR: Could not find the specified text in {file_path}"
            new_content = content.replace(old, new, 1)
            file_path.write_text(new_content)
            return f"OK: Replaced text in {file_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def analyze_issue(url: str) -> str:
        """Run RAG-powered analysis on a GitHub issue. Returns root cause, affected files, and proposed solution.

        Args:
            url: Full GitHub issue URL (e.g. https://github.com/owner/repo/issues/123).
        """
        try:
            analysis_service = services["analysis"]
            result = _run_async(analysis_service.analyze_github_issue(url))
            if result.success:
                return json.dumps(result.analysis, indent=2, default=str)
            else:
                return f"Analysis failed: {result.error_message}"
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def generate_patches(issue_body: str, repo_full_name: str) -> str:
        """Generate AI-suggested code patches for an issue.

        Args:
            issue_body: Description of the issue/bug to fix.
            repo_full_name: Repository in owner/repo format.
        """
        try:
            patch_service = services["patch"]
            result = _run_async(patch_service.generate_code_patch(issue_body, repo_full_name))
            return json.dumps({
                "summary": result.summary_of_changes,
                "files": [
                    {"path": fp.file_path, "changes": fp.changes}
                    for fp in (result.files_to_update or [])
                ],
            }, indent=2, default=str)
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def search_codebase(repo_name: str, query: str) -> str:
        """Semantic search over the ingested codebase using ChromaDB.

        Args:
            repo_name: Repository in owner/repo format.
            query: Natural language search query.
        """
        try:
            from langchain_chroma import Chroma

            collection_name = config.get_collection_name(repo_name, "code")
            embeddings = services["embedding"].get_embeddings()
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=config.chroma_persist_dir,
            )
            docs = vectorstore.similarity_search(query, k=5)
            results = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                results.append(f"--- {source} ---\n{doc.page_content[:1000]}")
            return "\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def search_learnings(repo_name: str, query: str) -> str:
        """Search stored learnings, patterns, and never-do rules for a repository.

        Args:
            repo_name: Repository in owner/repo format.
            query: Natural language search query.
        """
        try:
            learning_service = services["learning"]
            results = learning_service.search_learnings(repo_name, query, n_results=5)
            if not results:
                return "No learnings found for this query."
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def ingest_repo(repo_name: str) -> str:
        """Ingest a repository (docs, code, issues, PRs) into the vector database for RAG.

        Args:
            repo_name: Repository in owner/repo format (e.g. owner/repo).
        """
        try:
            ingestion_service = services["ingestion"]
            result = _run_async(ingestion_service.start_repository_ingestion(repo_name))
            if result.success:
                return f"Ingestion completed: {result.documents_stored} documents stored in {result.collection_name or repo_name}"
            else:
                return f"Ingestion failed: {result.error_message}"
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def get_repo_status(repo_name: str) -> str:
        """Check if a repository has been ingested and its ingestion status.

        Args:
            repo_name: Repository in owner/repo format.
        """
        try:
            state_manager = services["state_manager"]
            status = state_manager.get_repository_status(repo_name)
            if status is None:
                return f"Repository '{repo_name}' has NOT been ingested yet."
            return json.dumps({
                "repo": repo_name,
                "status": status.overall_status.value if hasattr(status.overall_status, 'value') else str(status.overall_status),
                "steps": {
                    str(step): {
                        "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                    }
                    for step, info in (status.steps or {}).items()
                } if status.steps else {},
                "error": status.error_message,
            }, indent=2, default=str)
        except Exception as e:
            return f"ERROR: {e}"

    @tool
    def show_diff(path: str = ".") -> str:
        """Show the current git diff of all changes in the working directory.

        Args:
            path: Optional path to limit diff scope. Defaults to all changes.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--", path],
                capture_output=True,
                text=True,
                cwd=workdir,
            )
            diff = result.stdout
            if not diff:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--", path],
                    capture_output=True,
                    text=True,
                    cwd=workdir,
                )
                diff = result.stdout
            return diff if diff else "No changes detected."
        except Exception as e:
            return f"ERROR: {e}"

    return [
        bash,
        read_file,
        write_file,
        edit_file,
        analyze_issue,
        generate_patches,
        search_codebase,
        search_learnings,
        ingest_repo,
        get_repo_status,
        show_diff,
    ]
