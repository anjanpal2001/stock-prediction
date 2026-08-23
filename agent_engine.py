import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from rag_engine import build_vector_store_from_ticker
from s3_utils import download_model_from_s3

load_dotenv()

@tool
def predict_stock_trend(ticker: str, last_price: float) -> str:
    """Predicts next-day price using the trained ML model for a ticker and its closing price."""
    ticker = ticker.upper().strip()
    filename = f"{ticker}_model.pkl"
    local_path = f"models/{filename}"
    
    if not os.path.exists(local_path):
        s3_key = f"models/{filename}"
        downloaded = download_model_from_s3(s3_key, local_path)
        if not downloaded:
            return f"No model found for {ticker} locally or on S3. Train the model first."
            
    model = joblib.load(local_path)
    input_data = pd.DataFrame([[last_price, last_price * 0.99, last_price * 1.01, last_price]], 
                              columns=['Close_lag1', 'Close_lag2', 'MA_5', 'MA_20'])
    prediction = model.predict(input_data)[0]
    return f"ML Model Predicted Next-Day Price for {ticker}: ${prediction:.2f}"

@tool
def query_live_financial_context(ticker: str, query: str) -> str:
    """Fetches real-time fundamental ratios and recent market news for a ticker."""
    try:
        retriever = build_vector_store_from_ticker(ticker)
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        return f"Live Context from Market Stream for {ticker}:\n{context}"
    except Exception as e:
        return f"Error retrieving real-time data for {ticker}: {str(e)}"

class FinancialAgent:
    def __init__(self):
        self.tools = {
            "predict_stock_trend": predict_stock_trend,
            "query_live_financial_context": query_live_financial_context
        }
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_with_tools = llm.bind_tools(list(self.tools.values()))

    def invoke(self, inputs: dict) -> dict:
        ticker = inputs.get("ticker", "")
        user_query = inputs.get("input", "")
        
        system_prompt = (
            "You are an AI Financial Analyst. Use your available tools to fetch "
            "live market fundamentals, news sentiment, and ML price predictions. "
            "Always formulate answers clearly and concisely."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Target Ticker: {ticker}\nUser Query: {user_query}")
        ]
        
        response = self.model_with_tools.invoke(messages)
        
        # Execute tool calls if requested by the LLM
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in self.tools:
                    tool_output = self.tools[tool_name].invoke(tool_args)
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            final_response = self.model_with_tools.invoke(messages)
            return {"output": final_response.content}
            
        return {"output": response.content}

def get_financial_agent():
    return FinancialAgent()