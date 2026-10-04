import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hermes_discovers_remote_mcp_before_building_agent():
    source = (ROOT / "app" / "hermes_chat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cognition = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_hermes_cognition"
    )
    calls = [
        node
        for node in ast.walk(cognition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    discovery = next(
        node
        for node in calls
        if node.func.id == "ensure_mcp_discovery_before_agent_build"
    )
    agent = next(node for node in calls if node.func.id == "AIAgent")
    assert discovery.lineno < agent.lineno

    toolsets = next(
        keyword.value
        for keyword in agent.keywords
        if keyword.arg == "enabled_toolsets"
    )
    assert any(
        isinstance(node, ast.Constant) and node.value == "mcp-reading_house"
        for node in ast.walk(toolsets)
    )


def test_core_declares_hermes_remote_mcp_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "mcp==2.0.0" in requirements.splitlines()
    assert "httpx2==2.7.0" in requirements.splitlines()


def test_reading_tooling_does_not_add_a_core_reading_proxy_route():
    routes = (ROOT / "app" / "api" / "routes.py").read_text(encoding="utf-8")
    assert '"/v1/reading/session"' in routes
    assert "/api/books" not in routes
    assert "/mcp" not in routes
