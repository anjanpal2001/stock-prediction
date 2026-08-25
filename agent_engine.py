import os
from dotenv import load_dotenv
import joblib
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import pandas as pd
from rag_engine import build_vector_store_from_ticker
from s3_utils import download_model_from_s3

load_dotenv()


# --- Tools Definition ---
import yfinance as yf

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

  # --- বাস্তব সাম্প্রতিক প্রাইস হিস্টরি আনা হচ্ছে, ফেক ভ্যালুর বদলে ---
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


# --- Agent Factory Function ---
def get_financial_agent():
  tools_map = {
      "predict_stock_trend": predict_stock_trend,
      "query_live_financial_context": query_live_financial_context,
  }

  llm = ChatGroq(
      model_name="qwen/qwen3.6-27b",
      temperature=0.0,
      api_key=os.getenv("GROQ_API_KEY"),
  )

  model_with_tools = llm.bind_tools(list(tools_map.values()))

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

    # Some reasoning-style models (Qwen thinking variants, etc.) put the
    # actual answer in additional_kwargs instead of .content.
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

    messages = [
        SystemMessage(
            content=(
                "You are an expert AI Financial Analyst. Use your available"
                " tools to fetch market data and ML price predictions. Once"
                " data is gathered, provide a clear, detailed final financial"
                " summary in plain text."
            )
        ),
        HumanMessage(
            content=f"Stock Ticker: {ticker}\nUser Query: {user_query}"
        ),
    ]

    last_text = ""

    # Tool calling loop
    for i in range(5):
      response = model_with_tools.invoke(messages)
      messages.append(response)

      # --- Debug logging: check your server console/terminal ---
      print(
          f"[agent iter {i}] content={response.content!r} "
          f"tool_calls={response.tool_calls} "
          f"additional_kwargs={getattr(response, 'additional_kwargs', {})}"
      )

      text_now = _extract_text(response)
      if text_now:
        last_text = text_now

      if not response.tool_calls:
        break

      for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in tools_map:
          try:
            tool_output = tools_map[tool_name].invoke(tool_args)
          except Exception as err:
            tool_output = f"Tool execution error: {err}"
        else:
          tool_output = f"Tool {tool_name} not found."

        print(f"[agent iter {i}] tool={tool_name} args={tool_args} -> {tool_output!r}")

        messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )
    else:
      # Loop exhausted all 5 iterations without a final break (model kept
      # calling tools). Log it so it's visible this is why output may be thin.
      print("[agent] WARNING: hit max tool-call iterations (5) without a final answer.")

    text_output = _extract_text(messages[-1]) or last_text

    if not text_output:
      text_output = (
          "The agent finished running but did not return any text. This"
          " usually means the model only returned tool calls, or the"
          " response text landed outside the expected field. Check the"
          " server console logs for the [agent iter] debug lines to see"
          " exactly what the model returned at each step."
      )

    return {"output": text_output}

  return run_agent