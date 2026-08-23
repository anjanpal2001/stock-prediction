from contextlib import asynccontextmanager
import os

# 1. Ensure models directory exists at startup
os.makedirs("models", exist_ok=True)
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


@app.post("/analyze")
def analyze_endpoint(
    request: Request, ticker: str = Form(...), query: str = Form(...)
):
  global agent_executor
  if not agent_executor:
    agent_executor = get_financial_agent()

  try:
    # run_agent expects a dict with "ticker" and "input" keys, NOT a string
    result = agent_executor({"ticker": ticker, "input": query})

    # run_agent always returns {"output": text}, but keep this defensive
    # in case the agent implementation changes later
    if isinstance(result, dict):
      output = (
          result.get("output")
          or result.get("response")
          or result.get("result")
          or str(result)
      )
    else:
      output = getattr(result, "content", str(result))

  except Exception as e:
    output = f"Execution Error: {str(e)}"

  return templates.TemplateResponse(
      request=request,
      name="index.html",
      context={"response": output, "ticker": ticker, "query": query},
  )


@app.post("/train-ml")
def train_ml_endpoint(request: Request, ticker: str = Form(...)):
  ticker = ticker.strip().upper()

  try:
    # Expected to train a model for the ticker and upload it to S3,
    # returning a human-readable status string
    status = train_and_save_model(ticker)
  except Exception as e:
    status = f"Training Error: {str(e)}"

  return templates.TemplateResponse(
      request=request,
      name="index.html",
      context={"train_status": status, "ticker": ticker},
  )


if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)