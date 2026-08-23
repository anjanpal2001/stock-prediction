import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv

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

# Point directly to the templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/train-ml")
def train_model_endpoint(request: Request, ticker: str = Form(...)):
    try:
        status = train_and_save_model(ticker=ticker)
    except Exception as e:
        status = f"Training Error: {str(e)}"
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"train_status": status}
    )

@app.post("/analyze")
def analyze_endpoint(request: Request, ticker: str = Form(...), query: str = Form(...)):
    global agent_executor
    if not agent_executor:
        agent_executor = get_financial_agent()
        
    try:
        result = agent_executor.invoke({"ticker": ticker, "input": query})
        output = result.get("output", "No response returned from agent.")
    except Exception as e:
        output = f"Execution Error: {str(e)}"
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"response": output}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)