import yfinance as yf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store_from_ticker(ticker:str):
    ticker=ticker.upper().strip()
    stock=yf.Ticker(ticker)
    # 1. Fetch live company profile and key financecials
    
    info=stock.info
    summary=info.get("longBusinessSummary","No company profile found.")
    market_cap=info.get("marketCap","N/A")
    trailing_pe=info.get("trailingPE","N/A")
    forward_pe=info.get("forwardPE","N/A")
    high_52=info.get("fiftyTwoWeekHigh","N/A")
    low_52=info.get("fiftyTwoWeekLow","N/A")
    
    # Fetch current market headlines
    news_record=stock.news or []
    news_text="\n".join([
        f"- Title: {item.get('title',' ')}"
        for item in news_record[:5]
    ])
    
    # 3.Create raw context
    full_text=f"""
    Stock Symbol: {ticker}
    Business Summary: {summary}
    Fundamentals:
    -Market Captalization: {market_cap}
    -Trailling P/E: {trailing_pe}
    -Forward P/E: {forward_pe}
    -52-Week High: {high_52}
    -52-Week Low: {low_52}
    
    Recent Headlines & Sentiment:
    {news_text}
    """
    doc=Document(page_content=full_text,metadata={"source":f"{ticker}_live_stream"})
    
    #4.Chunk text and generative vector index
    splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=60)
    docs_split=splitter.split_documents([doc])
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(docs_split,embeddings)
    return vectorstore.as_retriever()
    