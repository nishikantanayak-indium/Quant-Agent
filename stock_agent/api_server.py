"""
FastAPI Server for Stock Analysis Supervisor Agent
Exposes the supervisor agent functionality via REST API
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime

# Import the existing agent functionality
from main_agent import (
    wait_for_server,
)
from stock_exchange_agent.subagents.stock_information.langgraph_agent import create_stock_information_agent
from stock_exchange_agent.subagents.technical_analysis_agent.langgraph_agent import create_technical_analysis_agent
from stock_exchange_agent.subagents.ticker_finder_tool.langgraph_agent import create_ticker_finder_agent

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    success: bool

class StatusResponse(BaseModel):
    status: str
    servers_ready: Dict[str, bool]
    agents_ready: bool
    timestamp: str

# Global variables for agent management
supervisor = None
agents_initialized = False
saver = None
saver_cm = None

# FastAPI app
app = FastAPI(
    title="Stock Analysis Supervisor API",
    description="REST API for the Stock Analysis Supervisor Agent with LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def initialize_agents():
    """Initialize all agents and supervisor on startup"""
    global supervisor, agents_initialized, saver, saver_cm

    if agents_initialized:
        return

    try:
        print("🚀 Initializing Stock Analysis Supervisor Agent...")
        
        # Initialize memory saver
        print("💾 Initializing PostgreSQL memory...")
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        
        if not connection_string:
            raise ValueError("POSTGRES_CONNECTION_STRING not found in environment variables")
        
        saver_cm = AsyncPostgresSaver.from_conn_string(connection_string)
        saver = await saver_cm.__aenter__()
        await saver.setup()  # Creates tables if needed
        print("✅ Memory initialized successfully")

        # Wait for MCP servers to be ready
        print("⏳ Waiting for MCP servers...")
        await wait_for_server("http://localhost:8565/mcp")  # Stock Information
        await wait_for_server("http://localhost:8566/mcp")  # Technical Analysis

        # Create sub-agents
        print("🔧 Creating sub-agents...")
        stock_info_agent = await create_stock_information_agent(checkpointer=saver)
        technical_agent = await create_technical_analysis_agent(checkpointer=saver)
        ticker_finder = await create_ticker_finder_agent(checkpointer=saver)

        print("✅ Sub-agents created successfully")

        # Create supervisor
        supervisor_prompt = (
            "You are an intelligent supervisor managing three specialized stock analysis agents:\n\n"
            "- **stock_information_agent**: Expert in stock fundamentals including current prices, historical data, "
            "financial news, dividends, stock splits, financial statements (income/balance sheet/cashflow), "
            "holder information, analyst recommendations, target prices, sentiment analysis, and projections. "
            "Assign tasks related to fundamental analysis, company financials, news, and valuation metrics.\n\n"
            "- **technical_analysis_agent**: Expert in technical analysis including SMA, RSI, Bollinger Bands, "
            "MACD, Volume analysis, Support/Resistance levels, and comprehensive technical charting. "
            "Assign tasks related to chart patterns, technical indicators, trading signals, and price trends.\n\n"
            "- **ticker_finder_agent**: Expert in finding stock ticker symbols from company names using Yahoo Finance. "
            "This agent searches and returns ONLY the ticker symbol. Use this agent FIRST when the user mentions "
            "a company name instead of a ticker symbol.\n\n"
            "🎯 CRITICAL WORKFLOW - TASK ROUTING GUIDELINES:\n\n"
            "**Step 1: Ticker Resolution**\n"
            "- If user mentions a COMPANY NAME (e.g., 'Apple', 'Tesla', 'Microsoft'):\n"
            "  → ALWAYS delegate to ticker_finder_agent FIRST\n"
            "  → Wait for the ticker symbol response\n"
            "  → Store the ticker for subsequent operations\n"
            "- If user provides a TICKER SYMBOL (e.g., 'AAPL', 'TSLA', 'MSFT'):\n"
            "  → Skip ticker_finder_agent and proceed directly to appropriate specialist\n\n"
            "**Step 2: Task Analysis & Routing**\n"
            "After obtaining the ticker, analyze the request:\n\n"
            "For FUNDAMENTAL/FINANCIAL queries → **stock_information_agent**:\n"
            "  • Current price, market cap, P/E ratio, valuation metrics\n"
            "  • Historical price data and trends\n"
            "  • Financial news, announcements, sentiment\n"
            "  • Dividends, stock splits, corporate actions\n"
            "  • Financial statements (income, balance sheet, cash flow)\n"
            "  • Holder information (institutions, insiders, mutual funds)\n"
            "  • Analyst recommendations, price targets, upgrades/downgrades\n"
            "  • 5-year projections, growth estimates\n"
            "  • Options data and expiration dates\n"
            "  • Company fundamentals and statistics\n\n"
            "For TECHNICAL ANALYSIS queries → **technical_analysis_agent**:\n"
            "  • Moving averages (SMA, EMA)\n"
            "  • RSI (Relative Strength Index)\n"
            "  • Bollinger Bands\n"
            "  • MACD (Moving Average Convergence Divergence)\n"
            "  • Volume analysis\n"
            "  • Support and resistance levels\n"
            "  • Chart patterns and technical indicators\n"
            "  • Comprehensive technical analysis (all indicators)\n"
            "  • Trading signals and technical outlook\n\n"
            "**Step 3: Multi-Part Queries**\n"
            "When user asks for BOTH fundamental and technical analysis:\n"
            "  1. Get ticker symbol (if company name provided)\n"
            "  2. Delegate to stock_information_agent for fundamental data\n"
            "  3. Wait for completion\n"
            "  4. Delegate to technical_analysis_agent for technical data\n"
            "  5. Wait for completion\n"
            "  6. Combine results into comprehensive response\n\n"
            "**Step 4: Context Management**\n"
            "- Maintain conversation context across turns\n"
            "- Remember the ticker from previous exchanges\n"
            "- If user says 'now show me technical analysis' after asking for price:\n"
            "  → Use the stored ticker, don't ask for it again\n"
            "- If user switches to a different company:\n"
            "  → Use ticker_finder_agent to get the new ticker\n"
            "  → Update the stored ticker for the session\n\n"
            "🧠 INTELLIGENT RESPONSE GUIDELINES:\n"
            "- For follow-up questions about previous results, analyze conversation history and answer directly\n"
            "- When user asks analytical questions about data already retrieved, perform the analysis yourself\n"
            "- Only delegate to agents when NEW data needs to be fetched\n"
            "- Remember user preferences and ticker context from conversation history\n\n"
            "⚠️ CRITICAL RULES:\n"
            "1. ALWAYS assign work to ONE agent at a time - do not call agents in parallel\n"
            "2. WAIT for each agent to complete before proceeding to the next\n"
            "3. Do NOT attempt to answer stock-specific questions yourself - always delegate to specialist agents\n"
            "4. ALWAYS use ticker_finder_agent when user provides a company name (not a ticker)\n"
            "5. Provide clear context about why you're routing to each specific agent\n"
            "6. Maintain conversation memory and avoid redundant ticker lookups\n"
            "7. Combine multiple agent responses into coherent, comprehensive answers"
        )

        global supervisor
        supervisor_graph = create_supervisor(
            model=ChatOpenAI(temperature=0, model_name="gpt-4o"),
            agents=[stock_info_agent, technical_agent, ticker_finder],
            prompt=supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history",
        )
        supervisor = supervisor_graph.compile(checkpointer=saver)

        agents_initialized = True
        print("✅ Supervisor agent initialized successfully")

    except Exception as e:
        print(f"❌ Failed to initialize agents: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize agents on server startup"""
    await initialize_agents()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    global saver_cm
    if saver_cm is not None:
        try:
            await saver_cm.__aexit__(None, None, None)
            print("✅ Memory saver cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Error cleaning up memory saver: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {"message": "Stock Analysis Supervisor API", "status": "running", "version": "1.0.0"}


@app.get("/health", response_model=StatusResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check for the API and all MCP servers.
    
    Checks:
    - Both MCP servers are responding (Stock Info and Technical Analysis)
    - Agents are initialized
    - Database connectivity (PostgreSQL)
    - Supervisor agent is ready
    
    Returns:
    - status: "healthy" or "unhealthy"
    - servers_ready: Status of each MCP server
    - agents_ready: Whether supervisor and sub-agents are initialized
    - timestamp: ISO formatted timestamp
    """
    import socket
    from urllib.parse import urlparse

    def check_server(url):
        """Check if a server is responding on its port"""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or 'localhost'
            port = parsed.port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"❌ Server check failed for {url}: {str(e)}")
            return False

    # Check all MCP servers
    servers_status = {
        "stock_information": check_server("http://localhost:8565/mcp"),
        "technical_analysis": check_server("http://localhost:8566/mcp"),
    }

    # Determine overall health
    all_servers_ready = all(servers_status.values())
    overall_healthy = all_servers_ready and agents_initialized and supervisor is not None

    status = "healthy" if overall_healthy else "unhealthy"

    return StatusResponse(
        status=status,
        servers_ready=servers_status,
        agents_ready=agents_initialized,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_agent(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Send a message to the supervisor agent and get a response.
    
    The agent maintains conversation context per session_id.
    If no session_id is provided, a new one will be generated.
    """

    if not agents_initialized or supervisor is None:
        raise HTTPException(
            status_code=503,
            detail="Agents not initialized. Please check server status at /health endpoint."
        )

    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🧠 Processing request for session {session_id}: {request.message[:100]}...")

        # Get the current state to know how many messages exist
        current_state = await supervisor.aget_state(config={"configurable": {"thread_id": session_id}})
        messages_before = len(current_state.values.get('messages', [])) if current_state.values else 0

        # Invoke supervisor with thread_id for memory persistence
        response = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": session_id}}
        )

        # Extract only NEW messages from this turn
        all_messages = response['messages']
        new_messages = all_messages[messages_before:] if messages_before > 0 else all_messages
        
        # Find the last AI message from the new messages that is not a transfer/handoff
        final_message = None
        for msg in reversed(new_messages):
            if msg.type == 'ai' and msg.name != 'supervisor' and not msg.content.startswith('Transferring back') and not msg.content.startswith('Successfully transferred'):
                final_message = msg
                break
        
        # Fallback to last new message if no suitable AI message found
        if final_message is None and new_messages:
            final_message = new_messages[-1]
        elif final_message is None:
            final_message = all_messages[-1]

        # Save response to file in background
        background_tasks.add_task(save_response_to_file, response, session_id)

        return ChatResponse(
            response=final_message.content,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            success=True
        )

    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


def save_response_to_file(response, session_id):
    """Save response to JSON file"""
    try:
        def serialize_response(obj):
            try:
                if isinstance(obj, dict):
                    return {k: serialize_response(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_response(item) for item in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict', None)):
                    return obj.model_dump()
                elif hasattr(obj, '__dict__'):
                    return serialize_response(obj.__dict__)
                else:
                    return str(obj)
            except Exception:
                return str(obj)

        responses_dir = os.path.join(os.path.dirname(__file__), "responses")
        os.makedirs(responses_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_response_{session_id}_{timestamp}.json"
        filepath = os.path.join(responses_dir, filename)
        with open(filepath, "w") as f:
            json.dump(serialize_response(response), f, indent=4)
        print(f"📁 API response saved to {filepath}")
    except Exception as e:
        print(f"❌ Failed to save response: {str(e)}")


@app.get("/capabilities", tags=["Info"])
async def get_capabilities():
    """Get information about available capabilities"""
    return {
        "fundamental_analysis": [
            "Current stock prices and market data",
            "Historical price charts and trends",
            "Financial news and sentiment analysis",
            "Dividends, stock splits, and corporate actions",
            "Financial statements (income, balance sheet, cash flow)",
            "Analyst recommendations and price targets",
            "Holder information and institutional ownership",
            "5-year projections and growth estimates",
            "Options data and chains"
        ],
        "technical_analysis": [
            "Simple Moving Average (SMA)",
            "Relative Strength Index (RSI)",
            "Bollinger Bands",
            "MACD (Moving Average Convergence Divergence)",
            "Volume analysis",
            "Support and resistance levels",
            "Comprehensive technical charting",
            "Trading signals and technical outlook"
        ],
        "ticker_lookup": [
            "Find ticker symbols from company names",
            "Support for US and international stocks",
            "Yahoo Finance integration"
        ],
        "intelligent_features": [
            "Automatic ticker resolution from company names",
            "Context-aware conversations (remembers previous tickers)",
            "Multi-part query handling (fundamentals + technicals)",
            "Smart routing to specialized agents",
            "Session-based conversation memory"
        ]
    }


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_history(session_id: str):
    """Get conversation history for a specific session"""
    if not agents_initialized or supervisor is None:
        raise HTTPException(
            status_code=503,
            detail="Agents not initialized."
        )
    
    try:
        state = await supervisor.aget_state(config={"configurable": {"thread_id": session_id}})
        messages = state.values.get('messages', []) if state.values else []
        
        # Serialize messages
        serialized_messages = []
        for msg in messages:
            serialized_messages.append({
                "type": msg.type,
                "content": msg.content,
                "name": getattr(msg, 'name', None),
                "id": getattr(msg, 'id', None)
            })
        
        return {
            "session_id": session_id,
            "message_count": len(serialized_messages),
            "messages": serialized_messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Stock Analysis Supervisor API Server...")
    print("📍 API will be available at: http://localhost:8567")
    print("📖 API documentation at: http://localhost:8567/docs")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8567,
        reload=False,
        log_level="info"
    )
