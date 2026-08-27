import os
from dotenv import load_dotenv
import joblib
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from rag_engine import build_vector_store_from_ticker
from s3_utils import download_model_from_s3

load_dotenv()


# --- Tools Definition (unchanged) ---
@tool
def predict_stock_trend(ticker: str, last_price: float = None) -> str:
  """Predicts next-day price using the trained ML model for a ticker.

  Args:
      ticker: The stock symbol (e.g., TCS.NS, AAPL).
      last_price: Optional. Ignored if real market data can be fetched;
          kept only as a fallback label for the response text.
  """
  ticker = ticker.upper().strip()
  filename = f"{ticker}_model.pkl"
  local_path = f"models/{filename}"

  if not os.path.exists(local_path):
    s3_key = f"models/{filename}"
    downloaded = download_model_from_s3(s3_key, local_path)
    if not downloaded:
      return (
          f"No model found for {ticker} locally or on S3. Train the model"
          " first."
      )

  model = joblib.load(local_path)

  hist = yf.download(ticker, period="3mo", progress=False)
  if isinstance(hist.columns, pd.MultiIndex):
    hist.columns = hist.columns.get_level_values(0)

  if hist.empty or len(hist) < 21:
    return f"Not enough recent price history for {ticker} to build real features."

  hist["MA_5"] = hist["Close"].rolling(window=5).mean()
  hist["MA_20"] = hist["Close"].rolling(window=20).mean()
  hist = hist.dropna()

  latest = hist.iloc[-1]
  close_lag1 = latest["Close"]
  close_lag2 = hist.iloc[-2]["Close"]
  ma_5 = latest["MA_5"]
  ma_20 = latest["MA_20"]

  input_data = pd.DataFrame(
      [[close_lag1, close_lag2, ma_5, ma_20]],
      columns=["Close_lag1", "Close_lag2", "MA_5", "MA_20"],
  )
  prediction = model.predict(input_data)[0]
  return (
      f"ML Model Predicted Next-Day Price for {ticker}: ${prediction:.2f} "
      f"(based on last close ${close_lag1:.2f} on {hist.index[-1].date()})"
  )


@tool
def query_live_financial_context(ticker: str, query: str) -> str:
  """Fetches real-time fundamental ratios and recent market news for a ticker.

  Args:
      ticker: The stock symbol (e.g., TCS.NS, AAPL).
      query: The specific financial question or news topic to retrieve.
  """
  try:
    retriever = build_vector_store_from_ticker(ticker)
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    return f"Live Context from Market Stream for {ticker}:\n{context}"
  except Exception as e:
    return f"Error retrieving real-time data for {ticker}: {str(e)}"


# --- Agent Factory Function (now LangGraph-based) ---
def get_financial_agent():
  llm = ChatGroq(
      model_name="qwen/qwen3.6-27b",   # verify this is a live Groq model ID
      temperature=0.0,
      api_key=os.getenv("GROQ_API_KEY"),
  )

  system_prompt = (
      "You are an expert AI Financial Analyst. Use your available tools to"
      " fetch market data and ML price predictions. Once data is gathered,"
      " provide a clear, detailed final financial summary in plain text."
  )

  graph = create_agent(
      model=llm,
      tools=[predict_stock_trend, query_live_financial_context],
      system_prompt=system_prompt,
  )

  def _extract_text(message) -> str:
    """Pulls plain text out of an AIMessage, handling list-content and
    reasoning-model responses where the real answer sometimes lands in
    additional_kwargs instead of .content."""
    content = message.content

    if isinstance(content, list):
      text = "".join(
          c.get("text", "") if isinstance(c, dict) else str(c)
          for c in content
      )
    else:
      text = str(content) if content else ""

    if not text.strip():
      kwargs = getattr(message, "additional_kwargs", {}) or {}
      text = (
          kwargs.get("reasoning_content")
          or kwargs.get("reasoning")
          or ""
      )
    return text.strip()

  def run_agent(inputs: dict) -> dict:
    ticker = inputs.get("ticker", "").strip().upper()
    user_query = inputs.get("input", "").strip()

    result = graph.invoke({
        "messages": [
            HumanMessage(content=f"Stock Ticker: {ticker}\nUser Query: {user_query}")
        ]
    })

    final_messages = result["messages"]
    text_output = ""
    # Walk backwards to find the last message with usable text
    for msg in reversed(final_messages):
      text_output = _extract_text(msg)
      if text_output:
        break

    if not text_output:
      text_output = (
          "The agent finished running but did not return any text. Check"
          " server logs / LangSmith trace to inspect the message history."
      )

    return {"output": text_output}

  return run_agent