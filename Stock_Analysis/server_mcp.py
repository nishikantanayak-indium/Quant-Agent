import pandas as pd
import yfinance as yf
from fastmcp import FastMCP
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import ta # Technical Analysis library
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from scipy.signal import argrelextrema
import asyncio
import traceback
import base64
import sys
import os
import base64
from google import genai
from google.genai import types

# Initialize FastMCP server
technicalanalysis_server = FastMCP(
    "technicalanalysis",
    instructions="""
    # Technical Trading Analysis MCP Server

    This server generates a comprehensive technical analysis chart for a given stock using historical data from Yahoo Finance.

    Avilable tool:
        - `get_stock_sma`: Calculate the Simple Moving Average (SMA) for a given ticker symbol, start date, and end date.
        - `get_stock_rsi`: Generate an RSI (Relative Strength Index) chart for a stock between given dates.
        - `get_stock_bollingerbands`: Calculate the Bollinger Bands (BB) for a given ticker symbol, start date, and end date.
        - `get_stock_macd`: Calculate the Moving Average Convergence Divergence (MACD) for a given ticker symbol, start date, and end date.
        - `get_stock_volume`: Calculate the Volume for a given ticker symbol, start date, and end date.
        - `get_stock_support_resistance`: Calculate the Support and Resistance levels for a given ticker symbol, start date, and end date.   
    """
    )

async def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Asynchronously downloads stock data for a given ticker and date range.
    Returns a DataFrame or raises an exception if download fails or is empty.
    """
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: yf.download(ticker, start=start_date, end=end_date))

        if df.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")
    
        if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}\n\nTraceback:\n{traceback.format_exc()}")
                
            
async def save_figure_as_base64(fig: go.Figure, filename: str, width: int = 1000, height: int = 600) -> dict:
    try:
        loop = asyncio.get_event_loop()    
        await loop.run_in_executor(None, lambda: fig.write_image(filename, width=width, height=height))
        full_path = os.path.abspath(filename)
        with open(full_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {"filename": full_path, "image_base64": encoded}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

            

# Indicators Simple Moving Average
@technicalanalysis_server.tool(
        name="",
        description="""Calculate the Simple Moving Average (SMA) for the user's given ticker in the given date range."
        Args:
                ticker:str 
                        the ticker symbol given by the user for getting Simple Moving Average (SMA) of he company.
                start_date:str
                        the start date given by the user for getting Simple Moving Average (SMA) of the company.
                end_date:str
                        the end date given by the user for getting Simple Moving Average (SMA) of the company.
        Returns:
                str
                        A message indicating whether the SMA was successfully calculated or not.     
        """
)
async def get_stock_sma(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Simple Moving Average (SMA) for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate SMA for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate SMA for a stock (format: 'YYYY-MM-DD')
        """
        try:
                df = await fetch_stock_data(ticker, start_date, end_date)
                

                close = df['Close']
                df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
                df['SMA_100'] = SMAIndicator(close, window=100).sma_indicator()
                df['SMA_200'] = SMAIndicator(close, window=200).sma_indicator()
                df['SMA_300'] = SMAIndicator(close, window=300).sma_indicator()

                
        
                fig = go.Figure()
                df.index = df.index.strftime('%Y-%m-%d')
                fig.add_trace(go.Scatter(x=df.index, y=close, name='Close', line=dict(color='white')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_100'], name='SMA 100', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_300'], name='SMA 300', line=dict(color='purple')))
                print(df.index)
                fig.update_layout(
                title=f"{ticker.upper()} - Simple Moving Averages",
                template='plotly_dark',
                xaxis_title='Date',
                yaxis_title='Price',
        
                height=600,
                width=1000
                
        )
        
                filename = f"{ticker}_sma_chart.png"
                result = await save_figure_as_base64(fig, filename) 
                if "error" in result:
                        return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}

                return {
                "ticker": ticker,
                "image_filename": result["filename"],
                "image_base64": result["image_base64"]
                }
                
        except Exception as e:
               return {"ticker": ticker, "error": str(e), "traceback": traceback.format_exc()}
      

@technicalanalysis_server.tool(
    name="get_stock_rsi",
    description="""
    Generate an RSI (Relative Strength Index) chart for a stock between given dates.

    Args:
        ticker: str - Stock symbol (e.g., "NFLX")
        start_date: str - Start date in 'YYYY-MM-DD'
        end_date: str - End date in 'YYYY-MM-DD'

    Output:
        Base64-encoded chart of RSI values with thresholds.
    """
)
async def get_stock_rsi(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Relative Strength Index (RSI) for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate RSI for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate RSI for a stock (format: 'YYYY-MM-DD')
        """
        
        df = await fetch_stock_data(ticker, start_date, end_date)
        close=df['Close']
        rsi = RSIIndicator(close=close, window=14)
        
        df['RSI'] = rsi.rsi()

        # Plotting
        fig = go.Figure()
        df.index = df.index.strftime('%Y-%m-%d')
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI (14)',
            line=dict(color='lime')
        ))
        
        # Overbought / Oversold lines
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                      line=dict(color="red", dash="dash"), name='Overbought')
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                      line=dict(color="blue", dash="dash"), name='Oversold')

        fig.update_layout(
            title=f"{ticker.upper()} - RSI (Relative Strength Index)",
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            yaxis=dict(range=[0, 100]),
            height=500,
            width=1000,
            
        )

        filename = f"{ticker}_rsi_chart.png"
        result= await save_figure_as_base64(fig, filename)
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
            "ticker": ticker,
            "filename": result["filename"],
            "image_base64": result["image_base64"]
        }

#Bollinger Bands – show volatility using upper and lower band
@technicalanalysis_server.tool(
        name="get_stock_bollingerbands",
        description="""Calculate the Bollinger Bands (BB) for the user's given ticker in the given date range."
        Args:
                ticker:str 
                        the ticker symbol given by the user for getting Bollinger Bands (BB) of he company.
                start_date:str
                        the start date given by the user for getting Bollinger Bands (BB) of the company.
                end_date:str
                        the end date given by the user for getting Bollinger Bands (BB) of the company.
        Returns:
                str
                        A message indicating whether the Bollinger Bands (BB) was successfully calculated or not.     
        """
)
async def get_stock_bollingerbands(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Bollingerband (bb) for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate BollingerBand (bb) for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate BollingerBand (bb) for a stock (format: 'YYYY-MM-DD')
        """
        df = await fetch_stock_data(ticker, start_date, end_date)
        close = df['Close']
        bb = BollingerBands(close=close, window=20, window_dev=2)
        
        df['Upper_BB'] = bb.bollinger_hband()
        df['Lower_BB'] = bb.bollinger_lband()
        df['SMA_20'] = bb.bollinger_mavg()

        # Bollinger Bands
        fig = go.Figure()
        df.index = df.index.strftime('%Y-%m-%d')
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Upper_BB'],
            name='Upper Band',
            line=dict(color='lightblue', dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['Lower_BB'],
            name='Lower Band',
            line=dict(color='lightblue', dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))

        fig.update_layout(
            title=f"{ticker.upper()} - Bollinger Bands",
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            width=1000
        )

        filename = f"{ticker}_bollingerbands_chart.png"
        result= await save_figure_as_base64(fig, filename)
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
                "ticker": ticker,
                "image_filename": result["filename"],
                "image_base64": result["image_base64"]
                }
                
#MACD (Moving Average Convergence Divergence) – shows trend changes.
@technicalanalysis_server.tool(name="get_stock_macd",
    description="""Calculate the Moving Average Convergence Divergence (MACD) for the user's given ticker in the given date range.
        Args:
                ticker:str
                        the ticker symbol given by the user for getting Moving Average Convergence Divergence (MACD) of he company.
                start_date:str
                        the start date given by the user for getting Moving Average Convergence Divergence (MACD) of the company.
                end_date:str
                        the end date given by the user for getting Moving Average Convergence Divergence (MACD) of the company.
                        Returns:str
                                A message indicating whether the Moving Average Convergence Divergence (MACD) was successfully calculated or not.
        """
)

async def get_stock_macd(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Moving Average Convergence Divergence (MACD) for a given ticker symbol.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate MACD for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate MACD for a stock (format: 'YYYY-MM-DD')
        """
        df = await fetch_stock_data(ticker, start_date, end_date)
        close = df['Close']
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        
        df.index = df.index.strftime('%Y-%m-%d')
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_signal'], name='MACD_signal',
        line=dict(color='blue')), row=3, col=1)
        
        fig.update_layout(
        title=f"{ticker.upper()} - MACD Analysis",
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='MACD',
        height=900,
        width=1400
        )
        
        filename = f"{ticker}_macd_chart.png"
        result= await save_figure_as_base64(fig, filename)
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
                "ticker": ticker,
                "image_filename": result["filename"],
                "image_base64": result["image_base64"]
                }

# Volume Trends
@technicalanalysis_server.tool(
        name="get_stock_volume",
        description="""Calculate the Volume for the user's given ticker in the given date range.
        Args:
                ticker:str
                        the ticker symbol given by the user for getting Volume of he company.
                start_date:str
                        the start date given by the user for getting Volume of the company.
                end_date:str
                        the end date given by the user for getting Volume of the company.
        """
)
async def get_stock_volume(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Volume for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate Volume for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate Volume for a stock (format: 'YYYY-MM-DD')
        """
        df = await fetch_stock_data(ticker, start_date, end_date)
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        
        df.index = df.index.strftime('%Y-%m-%d')
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color='orange', opacity=0.4), row=4, col=1)
        
        fig.update_layout(
              title=f"{ticker.upper()} - Volume Trend (with 20-day SMA)",
             template='plotly_dark',
             xaxis_title='Date',
             yaxis_title='Volume',
             height=900,
             width=1400
         )
        
        filename = f"{ticker}_volume_trend.png"
        result= await save_figure_as_base64(fig, filename)
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
                "ticker": ticker,
                "image_filename": result["filename"],
                "image_base64": result["image_base64"]
                }

# Support/Resistance (using local minima/maxima)
@technicalanalysis_server.tool( 
        name="get_stock_support_resistance",
        description="""Calculate the Support and Resistance levels for the user's given ticker in the given date range.
        Args:
                ticker:str
                        the ticker symbol given by the user for getting Support and Resistance levels of he company.
                start_date:str
                        the start date given by the user for getting Support and Resistance levels of the company.
                end_date:str
                        the end date given by the user for getting Support and Resistance levels of the company.
                """
)
async def get_stock_support_resistance(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the Support and Resistance levels for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate Support and Resistance levels for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate Support and Resistance levels for a stock (format: 'YYYY-MM-DD')
        """
        df = await fetch_stock_data(ticker, start_date, end_date)
        n = 10  # sensitivity
        df['Support'] = df['Low'][argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]
        df['Resistance'] = df['High'][argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]
        
        
        df.index = df.index.strftime('%Y-%m-%d')
        fig = go.Figure()

        fig.add_trace(go.Scatter(
        x=df.index, y=df['Support'], mode='markers',
        name='Support', marker=dict(color='cyan', size=6, symbol='triangle-down')))
        fig.add_trace(go.Scatter(
        x=df.index, y=df['Resistance'], mode='markers',
        name='Resistance', marker=dict(color='red', size=6, symbol='triangle-up')))

        fig.update_layout(
        title=f'{ticker} Technical Analysis )',
        height=1000,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price'
        
)
        
        filename = f"{ticker}_support_resistance.png"
        #return await save_figure_as_base64(fig, filename)  
        result = await save_figure_as_base64(fig,filename )
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
            "ticker": ticker,
            "filename": result["filename"],
            "image_base64": result["image_base64"]
        }

        
@technicalanalysis_server.tool(
    name="get_all_technical_analysis",
    description="""
    Generate the SMA,RSI (Relative Strength Index),Volume,Support and resistance,MACD and BollingerBands charts for a stock between given dates.

    Args:
        ticker: str - Stock symbol (e.g., "NFLX")
        start_date: str - Start date in 'YYYY-MM-DD'
        end_date: str - End date in 'YYYY-MM-DD'

    Output:
        Base64-encoded chart of SMA,RSI (Relative Strength Index),Volume,Support and resistance,MACD,BollingerBands values with thresholds.
    """
)
async def get_all_technical_analysis(ticker: str, start_date: str, end_date: str) -> dict:
        """Get the SMA,RSI (Relative Strength Index),Volume,Support and resistance,MACD and BollingerBands charts  for a given ticker symbol, start date, and end date.
           Args: str
                ticker: The ticker symbol of the stock
                start_date: The start date to calculate SMA,RSI (Relative Strength Index),Volume,Support and resistance,MACD and BollingerBands  for a stock (format: 'YYYY-MM-DD')
                end_date: The end date to calculate SMA,RSI (Relative Strength Index),Volume,Support and resistance,MACD and BollingerBands for a stock (format: 'YYYY-MM-DD')
        """
        get_stock_sma(ticker, start_date, end_date)
        df = yf.download(ticker, start=start_date, end=end_date)

# Drop multilevel column index if present
        if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

# Indicators Simple Moving Average
        close = df['Close']
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_100'] = SMAIndicator(close, window=100).sma_indicator()
        df['SMA_200'] = SMAIndicator(close, window=200).sma_indicator()
        df['SMA_300'] = SMAIndicator(close, window=300).sma_indicator()

#Relative Strength Index – shows if the stock is overbought or oversold
        df['RSI'] = RSIIndicator(close, window=14).rsi()

#Bollinger Bands – show volatility using upper and lower band
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['Upper_BB'] = bb.bollinger_hband()
        df['Lower_BB'] = bb.bollinger_lband()

#MACD (Moving Average Convergence Divergence) – shows trend changes.
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

# Volume Trends
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

# Support/Resistance (using local minima/maxima)
        n = 10  # sensitivity
        df['Support'] = df['Low'][argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]
        df['Resistance'] = df['High'][argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]

# Subplots
        fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        subplot_titles=[
        f'{ticker} Price & SMAs',
        'RSI Indicator',
        'MACD',
        'Volume Trends'
        ]
        )


        df.index = df.index.strftime('%Y-%m-%d')
# Candlestick
        fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candlestick'
        ), row=1, col=1)



# Bollinger Bands
        fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_BB'], name='Upper BB',
        line=dict(color='lightblue', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_BB'], name='Lower BB',
        line=dict(color='lightblue', dash='dot')), row=1, col=1)

# RSI
        fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='lime', width=1)), row=2, col=1)

# Volume
        fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color='red', opacity=0.4), row=4, col=1)

        '''fig.add_trace(go.Scatter(
         x=df.index, y=df['Volume_SMA_20'], name='Vol SMA 20',
        line=dict(color='yellow', width=1)), row=4, col=1)'''

# SMAs
        for sma_col, color in zip(['SMA_20', 'SMA_100', 'SMA_200', 'SMA_300'],['orange', 'blue', 'green', 'purple']):
                fig.add_trace(go.Scatter(
                x=df.index, y=df[sma_col], name=sma_col,
                line=dict(color=color, width=1.5)), row=1, col=1)

        # Support and Resistance
        fig.add_trace(go.Scatter(
        x=df.index, y=df['Support'], mode='markers',
        name='Support', marker=dict(color='cyan', size=6, symbol='triangle-down')),
        row=1, col=1)
        fig.add_trace(go.Scatter(
        x=df.index, y=df['Resistance'], mode='markers',
        name='Resistance', marker=dict(color='red', size=6, symbol='triangle-up')),
        row=1, col=1)


# MACD
        fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='orange')), row=3, col=1)
        '''fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_signal'], name='MACD Signal',
        line=dict(color='lightblue')), row=3, col=1)'''

# Final layout
        fig.update_layout(
        title=f'{ticker} Technical Analysis (SMA, RSI, MACD, Volume, S/R)',
        height=1000,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        
        width=1200
        )
        
        filename = f"{ticker}_technical_analysis.png"
        result = await save_figure_as_base64(fig,filename )
        if "error" in result:
                return {"ticker": ticker, "error": result["error"], "traceback": result["traceback"]}
        else:
                return {
            "ticker": ticker,
            "filename": result["filename"],
            "image_base64": result["image_base64"]
        }

@technicalanalysis_server.tool(
    name="get_chart_summary",
    description="""
    Generate the detailed summary of the generated chart.

    Args:
        File_path: str - The path to the chart image file.

    Output:
        The summary of the chart, including technical indicators and patterns.
    """
)
async def get_chart_summary(file_path:str) -> dict:
        """This tool reads a Base64-encoded stock chart image from the provided file_path,
        sends it to Google's Gemini 2.5 Pro multimodal model, and returns a detailed,
        human-friendly technical analysis.
        Args:
                file_path (str): The path of the chart.
        Returns:
                dict: a dictionary containing the summary of the technical analysis.
        """
        with open(file_path, "rb") as f:
                image_bytes = f.read()
        
        img_b64 = image_bytes
        mime = "image/png"
        model="gemini-2.5-flash-lite"

        img_bytes = base64.b64decode(img_b64)
        if len(img_bytes) > 7 * 1024 * 1024:
                print("Warning: image >7MB, consider using Files API for better reliability.")

        system_prompt = (
                "You are an expert financial technical analyst. Produce a structured, "
                "plain-language summary for retail investors. Include headings: "
                "Overview, Indicators Detected, Interpretation, Key Levels, "
                "Trade Ideas (conservative/moderate/aggressive), Confidence & Risks, "
                "Plain-English Takeaway."
        )

        user_prompt = (
                f"Analyze the attached chart image containing all the technical indicators. "
                "1) Identify indicators shown (e.g., SMA, EMA, Bollinger Bands, MACD, RSI). "
                "2) Explain each indicator's current signal. "
                "3) List key price levels visible. "
                "4) Propose 3 trade ideas (entry, stop-loss, target, rationale). "
                "5) Give confidence level and one-sentence non-technical takeaway."
        )

        client = genai.Client(api_key="AIzaSyBuaXsBs7KVx-lH0BwHgjE3KbtPlnZFJQc")

        contents = [
                {
                "role": "user",
                "parts": [
                        {"text": system_prompt},
                        {"inline_data": {"mime_type": mime, "data": img_b64}},  # Base64 string
                        {"text": user_prompt}
                ]
                }
        ]

        resp = client.models.generate_content(
                model=model,
                contents=contents
        )
        summary = resp.text if hasattr(resp, "text") else resp.candidates[0].content.parts[0].text
        return {
            "summary": summary
            }
    
if __name__ == "__main__":
        import asyncio
        if sys.platform.startswith("win"):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        print("Launching TradingView MCP Server...")
        technicalanalysis_server.run(transport="streamable-http", port=8349)