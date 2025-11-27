import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Initialize client
llm = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Chroma (persistent in ./rag_store)
chroma_client = chromadb.PersistentClient(path="./rag_store")

# Create (or get) collection
collection = chroma_client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )
)

def embed_document(file_path: str):
    """Loads and embeds the document into Chroma."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunks = []
    chunk_size = 700
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    # Generate IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Insert into Chroma
    collection.add(ids=ids, documents=chunks)
    print(f"Embedded {len(chunks)} chunks from: {file_path}")

def retrieve(query: str, k: int = 3):
    """Retrieve top-k relevant chunks."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    return "\n\n".join(docs)

def answer_with_rag(query: str):
    """RAG pipeline: retrieve + generate answer."""
    context = retrieve(query)

    prompt = f"""
You are an assistant using RAG. Use the retrieved context to answer the user.
If context is insufficient, say so.

CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== RAG System Using ChromaDB ===")
    file_path = input("Enter document file path to embed: ")

    embed_document(file_path)

    print("\nDocument embedded. Type your questions.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Query > ")
        if q.lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break

        answer = answer_with_rag(q)
        print("\n--- ANSWER ---")
        print(answer)
        print("--------------\n")
