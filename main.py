from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import uvicorn

from agent_engine import get_financial_agent
from train_ml import train_and_save_model

load_dotenv()

agent_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global agent_executor
  agent_executor = get_financial_agent()
  yield


app = FastAPI(title="FinAgent Platform", lifespan=lifespan)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
  return templates.TemplateResponse(request, "index.html", {})


@app.post("/train-ml")
def train_model_endpoint(request: Request, ticker: str = Form(...)):
  try:
    status = train_and_save_model(ticker=ticker)
  except Exception as e:
    status = f"Training Error: {str(e)}"

  return templates.TemplateResponse(
      request, "index.html", {"train_status": status, "ticker": ticker}
  )


@app.post("/analyze")
def analyze_endpoint(
    request: Request, ticker: str = Form(...), query: str = Form(...)
):
  global agent_executor
  if not agent_executor:
    agent_executor = get_financial_agent()

  full_prompt = f"Analyze stock ticker {ticker.upper()}. Query: {query}"

  try:
    # Pass as a unified input prompt
    result = agent_executor.invoke({"input": full_prompt})

    print("\n--- RAW AGENT OUTPUT ---")
    print("DATA:", result)
    print("-------------------------\n")

    output = ""
    if isinstance(result, dict):
      output = result.get("output", "")
      # Fallback if output key is empty string
      if not output and "intermediate_steps" in result:
        steps = result["intermediate_steps"]
        if steps:
          output = str(steps[-1][1])

    if not output:
      output = "Agent executed the request but did not return text. Please check the model system prompt."

  except Exception as e:
    output = f"Execution Error: {str(e)}"
    print("Execution Error:", str(e))

  return templates.TemplateResponse(
      request,
      "index.html",
      {"response": output, "ticker": ticker, "query": query},
  )

if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)