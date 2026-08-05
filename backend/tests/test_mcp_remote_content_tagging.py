"""Tests for the MCP remote-content metadata tag.

MCP tool results are third-party, attacker-influenceable content, so MCP
registration tags every tool with ``deerflow_mcp_remote_content`` by default
(see ``get_mcp_tools``). A fully trusted local server can opt out via the
per-server ``sanitize_tool_results: false`` option in extensions_config.json.
These tests pin the default tagging, the opt-out, and the metadata helpers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from deerflow.config.extensions_config import McpServerConfig
from deerflow.mcp.tools import get_mcp_tools
from deerflow.tools.mcp_metadata import (
    MCP_TOOL_REMOTE_CONTENT_METADATA_KEY,
    is_remote_content_tool,
    tag_mcp_remote_content,
)


def _fetch_url_tool() -> StructuredTool:
    return StructuredTool.from_function(lambda: "page body", name="fetch_url", description="fetch a remote page")


def _run_get_mcp_tools(server_cfg: McpServerConfig | None):
    """Run get_mcp_tools() with a mocked MCP client returning one tool."""
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[_fetch_url_tool()])

    extensions = MagicMock(
        model_extra={},
        get_enabled_mcp_servers=MagicMock(return_value={}),
        mcp_servers={} if server_cfg is None else {"test-server": server_cfg},
    )

    with (
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=mock_client),
        patch("deerflow.config.extensions_config.ExtensionsConfig.from_file", return_value=extensions),
        patch("deerflow.mcp.tools.build_servers_config", return_value={"test-server": {}}),
        patch("deerflow.mcp.tools.get_initial_oauth_headers", new_callable=AsyncMock, return_value={}),
        patch("deerflow.mcp.tools.build_oauth_tool_interceptor", return_value=None),
    ):
        return asyncio.run(get_mcp_tools())


class TestRegistrationTagging:
    def test_mcp_tool_tagged_as_remote_content_by_default(self):
        tools = _run_get_mcp_tools(McpServerConfig(enabled=True))
        assert len(tools) == 1
        assert is_remote_content_tool(tools[0]) is True

    def test_mcp_tool_tagged_when_server_config_missing(self):
        # A server present in build_servers_config but absent from
        # extensions_config.mcp_servers must not lose protection.
        tools = _run_get_mcp_tools(None)
        assert len(tools) == 1
        assert is_remote_content_tool(tools[0]) is True

    def test_sanitize_tool_results_false_opts_out(self):
        tools = _run_get_mcp_tools(McpServerConfig(enabled=True, sanitize_tool_results=False))
        assert len(tools) == 1
        assert is_remote_content_tool(tools[0]) is False


class TestMetadataHelpers:
    def test_tag_and_predicate_roundtrip(self):
        tool = _fetch_url_tool()
        assert is_remote_content_tool(tool) is False
        tag_mcp_remote_content(tool)
        assert is_remote_content_tool(tool) is True
        assert tool.metadata[MCP_TOOL_REMOTE_CONTENT_METADATA_KEY] is True

    def test_tag_preserves_existing_metadata(self):
        tool = _fetch_url_tool()
        tool.metadata = {"deerflow_mcp": True}
        tag_mcp_remote_content(tool)
        assert tool.metadata["deerflow_mcp"] is True
        assert tool.metadata[MCP_TOOL_REMOTE_CONTENT_METADATA_KEY] is True

    def test_predicate_handles_none_tool(self):
        assert is_remote_content_tool(None) is False
