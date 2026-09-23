# Create the folders
import os

os.makedirs("/content/support_assistant/docs", exist_ok=True)

print("Folder created successfully!")

# Install all required libraries
!pip install -q chromadb sentence-transformers langgraph fastapi uvicorn pydantic

# Create the 8 documents
docs = {
    "doc_01.txt": """Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.""",

    "doc_02.txt": """Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.""",

    "doc_03.txt": """Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.""",

    "doc_04.txt": """Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.""",

    "doc_05.txt": """Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.""",

    "doc_06.txt": """If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.""",

    "doc_07.txt": """Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.""",

    "doc_08.txt": """Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."""
}

for filename, content in docs.items():
    path = f"/content/support_assistant/docs/{filename}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("All 8 documents created successfully!")

# Check the 8 files
import os

files = sorted(os.listdir("/content/support_assistant/docs"))

print("Documents:")
for file in files:
    print(file)

print("\nTotal documents:", len(files))

# Load the embedding model
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

# Create ChromaDB
import chromadb

chroma_client = chromadb.PersistentClient(
    path="/content/support_assistant/chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policy_corpus",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB collection created!")

# Insert the 8 documents into ChromaDB
from pathlib import Path

docs_path = Path("/content/support_assistant/docs")

documents = []
ids = []
metadatas = []

for file_path in sorted(docs_path.glob("doc_*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()

    documents.append(text)
    ids.append(file_path.stem)
    metadatas.append({
        "document_id": file_path.stem,
        "filename": file_path.name
    })

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Documents inserted into ChromaDB!")
print("Total documents:", collection.count())

# Test retrieval
query = "How much does delivery cost?"

query_embedding = embedding_model.encode(
    [query],
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("Retrieved documents:")

for i in range(len(results["ids"][0])):
    print("\nID:", results["ids"][0][i])
    print("Distance:", results["distances"][0][i])
    print("Text:", results["documents"][0][i][:200])

# Create the Pydantic output
from pydantic import BaseModel, Field
from typing import List


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AskRequest(BaseModel):
    query: str

# Create the LangGraph state
from typing import TypedDict, List


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float

# Create the structured prompt
STRUCTURED_PROMPT = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the Zepto policy information retrieved from the document corpus.

TASK:
Answer the user's question using the retrieved context.

FORMAT:
Return a JSON object with:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and easy to understand.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the retrieved Zepto policy context.
Do not invent policy details.

FEW-SHOT EXAMPLE:

Question:
What is the delivery fee for orders below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.

Answer:
{
  "answer": "Orders below INR 149 have a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
"""

# Set Mock Mode
import os

os.environ["MOCK_LLM"] = "1"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

print("MOCK_LLM =", MOCK_LLM)

# Create classify_intent node
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy"
    else:
        intent = "general"

    return {
        **state,
        "intent": intent
    }

#Testing
print(
    classify_intent({
        "query": "What is the delivery fee?"
    })
)

print(
    classify_intent({
        "query": "What is machine learning?"
    })
)

# Create retrieve_and_answer node
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_ids = results["ids"][0]
    retrieved_documents = results["documents"][0]

    top_chunk = retrieved_documents[0]

    snippet = top_chunk[:200]

    answer = f"Based on the retrieved context: {snippet}"

    return {
        **state,
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 1.0
    }

# Create direct_answer node
def direct_answer(state: GraphState) -> GraphState:

    answer = "I can only answer questions about Zepto policies right now."

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }

# Create the routing function
def route_by_intent(state: GraphState) -> str:

    if state["intent"] == "policy":
        return "retrieve_and_answer"

    return "direct_answer"

# Build LangGraph
from langgraph.graph import StateGraph, START, END


graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()

print("LangGraph compiled successfully!")

# Test the complete LangGraph
result = graph.invoke({
    "query": "How much does delivery cost?"
})

print(result)

# Test general question
result = graph.invoke({
    "query": "What is artificial intelligence?"
})

print(result)

# Validate the final output with Pydantic
result = graph.invoke({
    "query": "What is the refund policy?"
})

final_response = AskResponse(
    answer=result["answer"],
    sources=result["sources"],
    confidence=result["confidence"]
)

print(final_response.model_dump_json(indent=2))

# Create FastAPI
from fastapi import FastAPI

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    result = graph.invoke({
        "query": request.query
    })

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

# Test FastAPI inside Colab
import threading
import uvicorn


def run_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860
    )


server_thread = threading.Thread(
    target=run_server,
    daemon=True
)

server_thread.start()

print("FastAPI server started on port 7860")

import requests

response = requests.post(
    "http://127.0.0.1:7860/ask",
    json={
        "query": "How much does delivery cost?"
    }
)

print(response.status_code)
print(response.json())

response = requests.post(
    "http://127.0.0.1:7860/ask",
    json={
        "query": "What is Python?"
    }
)

print(response.status_code)
print(response.json())

# Create requirements.txt
requirements = """chromadb
sentence-transformers
langgraph
fastapi
uvicorn
pydantic
langchain-groq
"""

with open(
    "/content/support_assistant/requirements.txt",
    "w"
) as f:
    f.write(requirements)

print("requirements.txt created successfully!")

print(
    open(
        "/content/support_assistant/requirements.txt"
    ).read()
)

# Create Dockerfile
dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY docs ./docs

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

with open(
    "/content/support_assistant/Dockerfile",
    "w"
) as f:
    f.write(dockerfile)

print("Dockerfile created successfully!")

print(
    open(
        "/content/support_assistant/Dockerfile"
    ).read()
)

from pathlib import Path

root = Path("/content/drive/MyDrive")

print("Colab notebooks found:\n")

for path in root.rglob("*.ipynb"):
    print("📓", path)

from pathlib import Path
import shutil

drive = Path("/content/drive/MyDrive/Colab Notebooks")
project = Path("/content/zepto-data-ai-platform")

# Create final project folders
(project / "data_pipeline").mkdir(parents=True, exist_ok=True)
(project / "analytics").mkdir(parents=True, exist_ok=True)

# Copy Module 1 notebook
shutil.copy2(
    drive / "data_pipeline.ipynb",
    project / "data_pipeline" / "data_pipeline.ipynb"
)

# Copy Module 2 notebooks
shutil.copy2(
    drive / "analytics_01_EDA .ipynb",
    project / "analytics" / "analytics_01_EDA.ipynb"
)

shutil.copy2(
    drive / "analytics_02_modeling.ipynb",
    project / "analytics" / "analytics_02_modeling.ipynb"
)

print("✅ Module 1 notebook copied")
print("✅ Module 2 EDA notebook copied")
print("✅ Module 2 modeling notebook copied")
print("\nProject location:")
print(project)

from pathlib import Path
import shutil

source = Path("/content/support_assistant")
project = Path("/content/zepto-data-ai-platform")
destination = project / "support_assistant"

# Copy the complete Support Assistant folder
if destination.exists():
    shutil.rmtree(destination)

shutil.copytree(source, destination)

print("✅ Module 3 copied successfully")
print("\nSupport Assistant files:")
for path in sorted(destination.rglob("*")):
    if path.is_file():
        print("  ", path.relative_to(destination))

from pathlib import Path

project = Path("/content/zepto-data-ai-platform")

print("📁 FINAL PROJECT STRUCTURE\n")

for path in sorted(project.rglob("*")):
    if path.is_file():
        print("   ", path.relative_to(project))

from pathlib import Path

source = Path("/content/support_assistant")

print("Files in original Support Assistant folder:\n")

for path in sorted(source.rglob("*")):
    if path.is_file():
        print(path.relative_to(source))

from pathlib import Path

print("\nSearching for titanic.csv in Colab and Google Drive...\n")

for root in [Path("/content"), Path("/content/drive/MyDrive")]:
    for path in root.rglob("titanic.csv"):
        print(path)

import json

notebook_path = "/content/drive/MyDrive/Colab Notebooks/Support_assistance.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Number of cells:", len(nb["cells"]))

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])

        if "FastAPI" in code or "app = FastAPI" in code:
            print("\nFOUND FASTAPI CODE IN CELL:", i)
            print("-"*50)
            print(code[:3000])   # show first 3000 chars