"""Test-only Inspect tasks for the real-GPU integration gate.

The agentic task forces the first model action to be the registered tool.  That
makes tool execution an integration invariant instead of relying on the model
to decide whether the prompt is sufficiently persuasive.
"""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolFunction, tool


@tool
def aiq_real_gpu_double():
    """A tiny deterministic tool used to prove real agent/tool execution."""

    async def execute(value: int) -> str:
        """Double an integer.

        Args:
            value: Integer to double.
        """
        return str(value * 2)

    return execute


@task
def forced_tool_task():
    """Force one real model tool call, execute it, then let the model finish."""
    return Task(
        dataset=[Sample(
            input=(
                "Use the aiq_real_gpu_double tool on the integer 2. "
                "After the tool returns, answer with only the result."
            ),
            target="4",
        )],
        solver=[
            use_tools(
                aiq_real_gpu_double(),
                tool_choice=ToolFunction(name="aiq_real_gpu_double"),
            ),
            generate(),
        ],
        scorer=match(),
    )
