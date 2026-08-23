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
@tool
def predict_stock_trend(ticker: str, last_price: float) -> str:
  """Predicts next-day price using the trained ML model for a ticker and its closing price.

  Args:
      ticker: The stock symbol (e.g., TCS.NS, AAPL).
      last_price: The latest known closing price of the stock as a float.
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
  input_data = pd.DataFrame(
      [[last_price, last_price * 0.99, last_price * 1.01, last_price]],
      columns=["Close_lag1", "Close_lag2", "MA_5", "MA_20"],
  )
  prediction = model.predict(input_data)[0]
  return f"ML Model Predicted Next-Day Price for {ticker}: ${prediction:.2f}"


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

    # Tool calling loop
    for _ in range(5):
      response = model_with_tools.invoke(messages)
      messages.append(response)

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

        messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )

    content = messages[-1].content
    if isinstance(content, list):
      text_output = "".join(
          [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
      )
    else:
      text_output = str(content)

    return {"output": text_output}

  return run_agent