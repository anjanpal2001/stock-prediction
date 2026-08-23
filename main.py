import os
from contextlib import asynccontextmanager
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
  return templates.TemplateResponse(
      name="index.html", context={"request": request}
  )


@app.post("/train-ml")
def train_model_endpoint(request: Request, ticker: str = Form(...)):
  try:
    status = train_and_save_model(ticker=ticker)
  except Exception as e:
    status = f"Training Error: {str(e)}"

  return templates.TemplateResponse(
      name="index.html",
      context={"request": request, "train_status": status, "ticker": ticker},
  )


@app.post("/analyze")
def analyze_endpoint(
    request: Request, ticker: str = Form(...), query: str = Form(...)
):
  global agent_executor
  if not agent_executor:
    agent_executor = get_financial_agent()

  try:
    result = agent_executor.invoke({"ticker": ticker, "input": query})
    if isinstance(result, dict):
      output = result.get("output", str(result))
    else:
      output = str(result)
  except Exception as e:
    output = f"Execution Error: {str(e)}"

  return templates.TemplateResponse(
      name="index.html",
      context={
          "request": request,
          "response": output,
          "ticker": ticker,
          "query": query,
      },
  )


if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)