"""MCP resource registration: Claude skill files exposed over ``skill://`` URIs.

Kept out of ``main.py`` so server setup stays thin. Call
:func:`register_resources` with the FastMCP instance to wire these up.
"""

from collections.abc import Callable

from fastmcp import FastMCP

from .paths import get_skills_dir

# uri -> (skill markdown filename, docstring/title)
_SKILLS = {
    "skill://fan-trap-prevention": (
        "fan-trap-prevention.md",
        "Fan-trap prevention guide - comprehensive patterns and solutions.",
    ),
    "skill://sql-best-practices": (
        "sql-best-practices.md",
        "SQL best practices - identifier qualification and common patterns.",
    ),
    "skill://chart-examples": (
        "chart-examples.md",
        "Chart generation examples - all chart types with complete examples.",
    ),
    "skill://analytical-workflow": (
        "analytical-workflow.md",
        "Complete analytical session workflow - optimal tool chain and best practices.",
    ),
}


def _read_skill(filename: str) -> str:
    """Read a skill markdown file, returning a helpful message if missing."""
    skills_path = get_skills_dir() / filename
    if skills_path.exists():
        return skills_path.read_text()
    return f"Skill not found. Please ensure .claude/skills/{filename} exists."


def _make_skill_loader(filename: str, title: str) -> Callable[[], str]:
    """Build a zero-argument resource function bound to one skill file.

    FastMCP treats any function parameter as a URI-template variable, so the
    registered function must take no arguments; the filename is captured here.
    """

    def _loader() -> str:
        return _read_skill(filename)

    # Unique name/doc so FastMCP registers each resource distinctly.
    _loader.__name__ = filename.replace("-", "_").replace(".md", "") + "_skill"
    _loader.__doc__ = title
    return _loader


def register_resources(mcp: FastMCP) -> None:
    """Register all skill resources on the given FastMCP server."""
    for uri, (filename, title) in _SKILLS.items():
        mcp.resource(uri)(_make_skill_loader(filename, title))
