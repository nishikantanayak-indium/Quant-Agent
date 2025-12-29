#!/bin/bash
set -e
echo "Starting MCP servers and API server..."
# Start Yahoo Finance MCP server (port 8565)
python yahoo-finance-mcp/server.py &
# Start Technical Analysis MCP server (port 8566)
python Stock_Analysis/server_mcp.py &
# Start Research MCP server (port 8567)
python research_mcp/server_mcp.py &
# Wait a bit for servers to start
sleep 5
# Start API server (port 8568)
python stock_agent/api_server.py

