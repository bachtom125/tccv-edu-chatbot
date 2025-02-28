import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import re
import os
from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
import asyncio
from typing import AsyncGenerator
from .utils import duckduckgo_search

class StreamingCallback(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put(token)

    async def on_llm_end(self, response, **kwargs):
        await self.queue.put(None)  # Stop signal

    async def token_generator(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

load_dotenv()
PINECONE_API = os.getenv("PINECONE_API_KEY")
OPENAI_API = os.getenv("OPENAI_API_KEY")
GOOGLE_API = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API)
chunk_index = pinecone.Index("education-file-chunks")
file_info_index = pinecone.Index("education-file-info")

# Use custom embeddings class
embedding_model = HuggingFaceEmbeddings()

chunk_vectorstore = PineconeVectorStore(
    index=chunk_index, embedding=embedding_model, text_key='text'
)
file_info_vectorstore = PineconeVectorStore(
    index=file_info_index, embedding=embedding_model, text_key='id'
)

llm_for_query = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=GOOGLE_API
)
llm_for_response = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=GOOGLE_API, streaming=True
)

NUM_FILE_TO_FETCH = 3
NUM_CHUNK_TO_FETCH_PER_FILE = 3

def query_vectorstore(vectorstore: PineconeVectorStore, user_query: str, top_k: int = 3, **kwargs):
    results = vectorstore.similarity_search(user_query, k=top_k, **kwargs)
    return results if results else "No relevant results found."

def query_for_file(user_query: str, top_k: int = 3):
    query_res = query_vectorstore(file_info_vectorstore, user_query, top_k)
    return [doc.page_content for doc in query_res]

def query_for_chunks(user_query: str, file_ids: list[str], top_k: int = 3):
    final_chunks = []
    for file_id in file_ids:
        query_res = query_vectorstore(chunk_vectorstore, user_query, top_k, filter={"file_path": file_id})
        final_chunks.extend([doc.page_content for doc in query_res])
    return final_chunks

def refine_query(user_query: str, past_messages: str):
    response = llm_for_query.predict(
        f"Constructed Query:\n{query_prompt_template.format(user_query=user_query, past_messages=past_messages)}"
    )
    refined_query = re.search(r"Constructed Query:\s*(.+)", response).group(1).strip()
    return refined_query

def fetch_vector_search_context(chunks: list[str]):
    return "\n".join(chunks)

def fetch_web_search_context(refined_query: str):
    return duckduckgo_search(refined_query, max_results=3)

def aggregate_context(vector_search_context: str, web_search_context: str):
    return f"{vector_search_context}\n\n**Additional Context from DuckDuckGo Web Search:**\n{web_search_context}"

def generate_response(context: str, user_query: str, refined_query: str, consulted_files: list[str], past_messages: str):
    response_prompt = response_generation_prompt.format(
        context=context,
        user_query=user_query,
        refined_query=refined_query,
        consulted_files=consulted_files,
        past_messages=past_messages
    )
    return response_prompt

async def stream_llm_response(response_prompt: str) -> AsyncGenerator[str, None]:
    sync_iterator = llm_for_response.stream(response_prompt)
    for chunk in sync_iterator:
        await asyncio.sleep(0)
        yield chunk.content

def run_query_pipeline(user_query: str, past_messages: str):
    refined_query = refine_query(user_query, past_messages)
    file_ids = query_for_file(refined_query, NUM_FILE_TO_FETCH)
    chunks = query_for_chunks(refined_query, file_ids, NUM_CHUNK_TO_FETCH_PER_FILE)
    vector_search_context = fetch_vector_search_context(chunks)
    web_search_context = fetch_web_search_context(refined_query)
    context = aggregate_context(vector_search_context, web_search_context)
    response_prompt = generate_response(context, user_query, refined_query, file_ids, past_messages)
    return response_prompt

print("Pipeline initialized")
