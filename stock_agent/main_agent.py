"""
Main LangGraph Supervisor Agent for Stock Analysis
---------------------------------------------------
Manages Stock Information, Technical Analysis, and Ticker Finder agents as specialized sub-agents.
Uses langgraph-supervisor to coordinate work between agents.
"""

import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv
from stock_exchange_agent.subagents.stock_information.langgraph_agent import create_stock_information_agent
from stock_exchange_agent.subagents.technical_analysis_agent.langgraph_agent import create_technical_analysis_agent
from stock_exchange_agent.subagents.ticker_finder_tool.langgraph_agent import create_ticker_finder_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
from datetime import datetime
import json

load_dotenv()


async def wait_for_server(url: str, timeout: int = 10):
    """Wait until the MCP server is ready to accept connections."""
    import time
    import socket
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"✅ MCP server is up at {url}")
                return True
        except:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"MCP server at {url} did not respond within {timeout} seconds")


async def main():
    """Main supervisor agent that coordinates stock analysis sub-agents."""
    
    print("🚀 Initializing Stock Analysis Supervisor Agent...")
    print("=" * 80)
    
    # Initialize memory saver
    print("💾 Initializing PostgreSQL memory...")
    connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
    
    if not connection_string:
        print("❌ ERROR: POSTGRES_CONNECTION_STRING not found in environment variables")
        print("Please set it in your .env file:")
        print('POSTGRES_CONNECTION_STRING="postgresql://user:password@localhost:5432/dbname"')
        return
    
    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
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
        supervisor_graph = create_supervisor(
            model=ChatOpenAI(temperature=0, model_name="gpt-4o"),
            agents=[stock_info_agent, technical_agent, ticker_finder],
            prompt=(
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
                "- For follow-up questions about previous results (e.g., 'how many?', 'what was the price?'), "
                "analyze conversation history and answer directly without re-running tools\n"
                "- When user asks analytical questions about data already retrieved, perform the analysis yourself "
                "(count, summarize, compare) instead of delegating\n"
                "- Only delegate to agents when NEW data needs to be fetched\n"
                "- Remember user preferences and ticker context from conversation history\n"
                "- Track what information has already been provided to avoid redundancy\n\n"
                "📋 EXAMPLE WORKFLOWS:\n\n"
                "**Example 1: Single Company, Single Request**\n"
                "User: 'What is Apple's current stock price?'\n"
                "You: \n"
                "  1. Delegate to ticker_finder_agent('Apple') → receives 'AAPL'\n"
                "  2. Delegate to stock_information_agent('AAPL', 'current price')\n"
                "  3. Present the price to user\n\n"
                "**Example 2: Ticker Provided, Technical Analysis**\n"
                "User: 'Show me RSI chart for TSLA'\n"
                "You:\n"
                "  1. Skip ticker_finder (already have TSLA)\n"
                "  2. Delegate to technical_analysis_agent('TSLA', 'RSI')\n"
                "  3. Present the technical analysis\n\n"
                "**Example 3: Multi-Part Query**\n"
                "User: 'Give me Microsoft's latest news and technical analysis'\n"
                "You:\n"
                "  1. Delegate to ticker_finder_agent('Microsoft') → 'MSFT'\n"
                "  2. Delegate to stock_information_agent('MSFT', 'news')\n"
                "  3. Wait for completion\n"
                "  4. Delegate to technical_analysis_agent('MSFT', 'comprehensive analysis')\n"
                "  5. Wait for completion\n"
                "  6. Combine and present both results\n\n"
                "**Example 4: Context Continuation**\n"
                "User: 'What is Tesla's price?'\n"
                "You: [Get TSLA, show price]\n"
                "User: 'Now show me the RSI'\n"
                "You:\n"
                "  1. Recall ticker from context (TSLA)\n"
                "  2. Delegate to technical_analysis_agent('TSLA', 'RSI')\n"
                "  3. Present RSI analysis\n\n"
                "**Example 5: Company Switch**\n"
                "User: 'What is Apple's price?'\n"
                "You: [Get AAPL, show price]\n"
                "User: 'How about Amazon?'\n"
                "You:\n"
                "  1. Delegate to ticker_finder_agent('Amazon') → 'AMZN'\n"
                "  2. Delegate to stock_information_agent('AMZN', 'current price')\n"
                "  3. Present Amazon's price\n\n"
                "⚠️ CRITICAL RULES:\n"
                "1. ALWAYS assign work to ONE agent at a time - do not call agents in parallel\n"
                "2. WAIT for each agent to complete before proceeding to the next\n"
                "3. Do NOT attempt to answer stock-specific questions yourself - always delegate to specialist agents\n"
                "4. ALWAYS use ticker_finder_agent when user provides a company name (not a ticker)\n"
                "5. Provide clear context about why you're routing to each specific agent\n"
                "6. Maintain conversation memory and avoid redundant ticker lookups\n"
                "7. Combine multiple agent responses into coherent, comprehensive answers\n\n"
                "Remember: You are the orchestrator. Your job is to understand user intent, manage the workflow, "
                "ensure proper routing, maintain context, and deliver comprehensive results by coordinating the "
                "specialized agents. You do not have direct access to stock data - you must delegate to the appropriate agents."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        )
        supervisor = supervisor_graph.compile(checkpointer=saver)
        
        print("\n" + "="*80)
        print("🤖 STOCK ANALYSIS SUPERVISOR AGENT - Ready for Commands")
        print("="*80)
        print("\n📋 What I can help you with:")
        print("\n📊 FUNDAMENTAL ANALYSIS:")
        print("  • Current stock prices and market data")
        print("  • Historical price charts and trends")
        print("  • Financial news and sentiment analysis")
        print("  • Dividends, stock splits, and corporate actions")
        print("  • Financial statements and company financials")
        print("  • Analyst recommendations and price targets")
        print("  • Holder information and institutional ownership")
        print("  • 5-year projections and growth estimates")
        
        print("\n📈 TECHNICAL ANALYSIS:")
        print("  • Moving averages (SMA, EMA)")
        print("  • RSI and momentum indicators")
        print("  • Bollinger Bands and volatility")
        print("  • MACD and trend analysis")
        print("  • Volume analysis")
        print("  • Support and resistance levels")
        print("  • Comprehensive technical charting")
        
        print("\n🔍 TICKER LOOKUP:")
        print("  • Find ticker symbols from company names")
        print("  • Support for US and international stocks")
        
        print("\n🤖 INTELLIGENT FEATURES:")
        print("  • Automatic ticker resolution from company names")
        print("  • Context-aware conversation (remembers previous tickers)")
        print("  • Multi-part query handling (fundamentals + technicals)")
        print("  • Smart routing to specialized agents")
        
        print("\nEnter your command (or 'quit' to exit): ")
        
        while True:
            try:
                user_input = input("\n>>> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🧠 Processing: {user_input}")
                print("-" * 50)
                
                # Get the current state to know how many messages exist
                current_state = await supervisor.aget_state(config={"configurable": {"thread_id": "main_thread"}})
                messages_before = len(current_state.values.get('messages', [])) if current_state.values else 0
                
                # Invoke supervisor with thread_id for memory persistence
                response = await supervisor.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": "main_thread"}}
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
                
                print("\n🤖 Response:")
                print(final_message.content)

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
                filename = f"response_{timestamp}.json"
                filepath = os.path.join(responses_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(serialize_response(response), f, indent=4)
                print(f"📁 Response saved to {filepath}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("💾 Memory saved successfully")


if __name__ == "__main__":
    asyncio.run(main())
