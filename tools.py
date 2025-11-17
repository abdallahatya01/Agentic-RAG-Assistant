from crewai.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_API_KEY

# === 1. Load PDF ===
loader = PyPDFLoader("data/attention_is_all_you_need.pdf")
docs = loader.load()

# === 2. Embeddings & Vector Store ===
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === 3. Re-Ranker Setup ===
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class SimpleCrossEncoderReranker:
    def __init__(self, model, top_n=2):
        self.model = model
        self.top_n = top_n

    def compress_documents(self, docs, query):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_n]]

compressor = SimpleCrossEncoderReranker(cross_encoder, top_n=2)

class RerankRetriever:
    def __init__(self, base_retriever, reranker):
        self.base_retriever = base_retriever
        self.reranker = reranker

    def invoke(self, query: str):
        docs = self.base_retriever.invoke(query)
        return self.reranker.compress_documents(docs, query)

compression_retriever = RerankRetriever(base_retriever, compressor)

@tool
def rag_tool(query: str) -> str:
    """Retrieve EXACT text from PDFs. No LLM. No summarization. Returns raw context."""
    retrieved_docs = compression_retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    return context if context.strip() else "No relevant context found in the knowledge base."

@tool
def web_search_tool(query: str) -> str:
    """Search the web using Tavily."""
    tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
    results = tavily_tool.invoke(query)
    return "\n\n".join([
        f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}"
        for res in results
    ])
