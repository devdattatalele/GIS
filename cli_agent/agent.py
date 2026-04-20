"""Core ReAct agent loop.

The AgentRunner drives the LLM in a tool-calling loop until the task
is complete or max iterations are reached.
"""

import re
import time
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from cli_agent.display import Display
from cli_agent.prompts import build_system_prompt
from cli_agent.tools import create_tools


PR_URL_PATTERN = re.compile(r"https://github\.com/[^\s]+/pull/\d+")


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    summary: str = ""
    pr_url: Optional[str] = None
    iterations: int = 0
    tool_count: int = 0
    duration: float = 0.0


def _extract_text_content(content) -> str:
    """Extract text from LLM response content.

    Handles both str (Gemini) and list-of-blocks (Claude) formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)
    return str(content)


class AgentRunner:
    """Autonomous ReAct agent that resolves GitHub issues."""

    def __init__(
        self,
        config,
        services: dict,
        workdir: str,
        max_iterations: int = 30,
        auto_pr: bool = False,
        verbose: bool = False,
        display=None,
    ):
        self.config = config
        self.services = services
        self.workdir = workdir
        self.max_iterations = max_iterations
        self.auto_pr = auto_pr
        self.display = display if display is not None else Display(verbose=verbose)
        self.tools = create_tools(config, services, workdir)
        self._tool_map = {t.name: t for t in self.tools}
        self._pr_url: Optional[str] = None
        self._tool_count = 0

    def run(self, issue_url: str) -> AgentResult:
        """Run the agent loop to resolve an issue.

        This is a synchronous method — it drives the LLM + tool loop.
        """
        start_time = time.time()

        self.display.show_banner()
        self.display.show_agent_start(issue_url, self.workdir)

        # Build LLM with tools bound
        llm = self.services["llm"].get_llm()
        llm_with_tools = llm.bind_tools(self.tools)

        # Initialize messages
        messages = [
            SystemMessage(content=build_system_prompt()),
            HumanMessage(content=f"Resolve this GitHub issue: {issue_url}"),
        ]

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.display.show_iteration_header(iteration, self.max_iterations)

            # Call LLM
            try:
                with self.display.show_thinking_spinner():
                    response = llm_with_tools.invoke(messages)
            except Exception as e:
                # One retry
                self.display.show_error(f"LLM call failed: {e}. Retrying...")
                time.sleep(2)
                try:
                    with self.display.show_thinking_spinner():
                        response = llm_with_tools.invoke(messages)
                except Exception as e2:
                    return AgentResult(
                        success=False,
                        summary=f"LLM call failed after retry: {e2}",
                        iterations=iteration,
                        tool_count=self._tool_count,
                        duration=time.time() - start_time,
                    )

            # Show reasoning text if present
            text = _extract_text_content(response.content)
            if text.strip():
                self.display.show_agent_reasoning(text)

            # If no tool calls, the agent is done
            if not response.tool_calls:
                self.display.show_agent_response(text)
                return AgentResult(
                    success=True,
                    summary=text,
                    pr_url=self._pr_url,
                    iterations=iteration,
                    tool_count=self._tool_count,
                    duration=time.time() - start_time,
                )

            # Process tool calls
            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", f"call_{self._tool_count}")

                self.display.show_tool_call(tool_name, tool_args)

                # PR creation guard
                if tool_name == "bash" and "gh pr create" in tool_args.get("command", ""):
                    if not self.auto_pr:
                        if not self.display.confirm_pr_creation():
                            result_str = "PR creation skipped by user."
                            messages.append(ToolMessage(
                                content=result_str,
                                tool_call_id=tool_call_id,
                            ))
                            self.display.show_tool_result(tool_name, result_str)
                            continue

                # Execute tool
                result_str = self._execute_tool(tool_name, tool_args)
                self._tool_count += 1

                # Capture PR URL from bash output
                if tool_name == "bash":
                    match = PR_URL_PATTERN.search(result_str)
                    if match:
                        self._pr_url = match.group(0)

                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tool_call_id,
                ))
                self.display.show_tool_result(tool_name, result_str)

        # Max iterations reached
        return AgentResult(
            success=False,
            summary="Max iterations reached without completion.",
            pr_url=self._pr_url,
            iterations=iteration,
            tool_count=self._tool_count,
            duration=time.time() - start_time,
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name, return result as string."""
        tool_fn = self._tool_map.get(name)
        if tool_fn is None:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self._tool_map.keys())}"
        try:
            result = tool_fn.invoke(args)
            return str(result) if result is not None else "OK"
        except Exception as e:
            return f"ERROR executing {name}: {e}"
