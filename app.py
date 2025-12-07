# app.py
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import traceback

# your project imports (keep them as you have them)
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *  # system_prompt should be defined here

load_dotenv()
app = Flask(__name__)

# --- load / validate env vars early ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not PINECONE_API_KEY:
    app.logger.warning("PINECONE_API_KEY not set. Pinecone calls will fail if attempted.")
if not OPENROUTER_API_KEY:
    app.logger.warning("OPENROUTER_API_KEY not set. LLM calls will fail if attempted.")

# keep these available for libraries that read from env
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# --- Initialize heavy objects once at startup (with safe guards) ---
try:
    embeddings = download_hugging_face_embeddings()
except Exception as e:
    app.logger.error("Failed to get embeddings: %s", e)
    embeddings = None

index_name = "medibot2"
docsearch = None
retriever = None
try:
    if embeddings is None:
        raise RuntimeError("Embeddings unavailable; cannot initialize Pinecone index.")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    app.logger.error("Failed to initialize PineconeVectorStore: %s", e)
    docsearch = None
    retriever = None

# --- LLM / RAG chain setup (safe-guarded) ---
llm = None
question_answer_chain = None
rag_chain = None
try:
    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        temperature=0.4,
        max_tokens=500,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    if retriever is None:
        raise RuntimeError("Retriever is None; cannot create retrieval chain.")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

except Exception as e:
    app.logger.error("Failed to create LLM/chain: %s", e)
    rag_chain = None


# --- Routes ---
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST", "GET"])
def chat():
    """
    Accepts either:
      - JSON body with {"message": "..." }
      - Form field 'msg' (older clients)
    Returns JSON: {"reply": "<text>"}
    """
    try:
        # prefer JSON (matches the chat.html script you have)
        data = request.get_json(silent=True) or {}
        msg = data.get("message") or request.form.get("msg") or request.args.get("msg")

        if not msg:
            return jsonify({"error": "No message provided."}), 400

        # If the RAG chain is not available, return a helpful error instead of throwing.
        if rag_chain is None:
            app.logger.error("RAG chain not initialized. Returning placeholder response.")
            return jsonify({"reply": "MediBot is starting up or misconfigured (RAG chain not ready)."}), 503

        # invoke chain (depending on the library the call might be .invoke or .run)
        response = rag_chain.invoke({"input": msg})
        # many chains return { "answer": "..."} or {"output_text": "..."} -- try a few fallbacks
        answer = None
        if isinstance(response, dict):
            answer = response.get("answer") or response.get("output_text") or response.get("result") or response.get("output")
        if not answer:
            # fallback to stringifying response
            answer = str(response)

        return jsonify({"reply": answer})

    except Exception as e:
        app.logger.error("Error processing /get: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "Internal server error - check server logs."}), 500


if __name__ == "__main__":
    # debug True is fine for local dev; remove for production.
    app.run(host="0.0.0.0", port=8080, debug=True)
