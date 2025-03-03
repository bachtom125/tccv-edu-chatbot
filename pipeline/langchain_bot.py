import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_pinecone import PineconeVectorStore
import re
import os
from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
import asyncio
from typing import AsyncGenerator
from .utils import duckduckgo_search
import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_pinecone import PineconeVectorStore
import re
import os
from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
import asyncio
from typing import AsyncGenerator
from pipeline.utils import duckduckgo_search

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
# HUGGINGFACE_API = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API)
chunk_index = pinecone.Index("education-file-chunks")
file_info_index = pinecone.Index("education-file-info")

# Use your custom embeddings class
embedding_model = HuggingFaceEmbeddings()

# Create a vector store for chunks
chunk_vectorstore = PineconeVectorStore(
    index=chunk_index, 
    embedding=embedding_model, 
    text_key='text'
)

# Create a vector store for file info (summaries)
file_info_vectorstore = PineconeVectorStore(
    index=file_info_index, 
    embedding=embedding_model, 
    text_key='file_path'
)

# Chain to use the LLM with the prompt
llm_for_query = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API
)

llm_for_response = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API,
    streaming=True  # Enable streaming
)

# CONFIG
NUM_FILE_TO_FETCH = 3
NUM_CHUNK_TO_FETCH_PER_FILE = 3

def query_vectorstore(vectorstore: PineconeVectorStore, user_query: str, top_k: int = 3, **kwargs):
    # Step 1: Query Pinecone
    results = vectorstore.similarity_search(user_query, k=top_k, **kwargs)
    if not results:
        return "No relevant results found."

    return results

def query_for_file(user_query: str, top_k: int = 3):
    """
    Query the vector store for file info (summaries) and return the top_k file_paths and file_urls.
    """
    query_res = query_vectorstore(file_info_vectorstore, user_query, top_k)

    file_info = [(doc.page_content, doc.metadata['url']) for doc in query_res]
    return file_info 

def query_for_chunks(user_query: str, file_ids: list[str], top_k: int = 3):
    """
    Given the target file_ids, query the vector store for top_k chunks FOR EACH FILE.
    """

    # Pass a filter to restrict the search to the identified files.
    final_chunks = []
    for file_id in file_ids:
        query_res = query_vectorstore(chunk_vectorstore, user_query, top_k, filter={"file_path": file_id})
        chunks = [doc.page_content for doc in query_res]
        final_chunks.extend(chunks)
    return final_chunks

def parse_input(messages):
    assistant_responses = []
    user_messages = []

    for msg in messages:
        role = msg["role"].capitalize()  # Capitalize role for formatting
        content = msg["content"]
        
        if role == "Assistant":
            assistant_responses.append(f"Assistant's Response: {content}")
            # parsed_output.append(f"Assistant's Response: {content}")
        elif role == "User":
            user_messages.append(f"User's Message: {content}")
            # parsed_output.append(f"User's Message: {content}")
    
    if len(assistant_responses) == 0:
      assistant_responses.append("")
    if len(user_messages) == 0:
      user_messages.append("User's Message: \"")

    return "\n".join(user_messages[:-1]) + f"\n{assistant_responses[-1]}", user_messages[-1][len("User's Message: "):]
    
# Define the prompt template
query_prompt_template = PromptTemplate(
    input_variables=["user_query", "past_messages"],
    template=(
        "You are an advanced query assistant with expertise in carbon credits fluent in both English and Vietnamese."
        "Given past messages between the user and the system, and user's current query, construct a meaningful, natural language, contextually complete representation of the current query that can be vectorized for semantic similarity search."
        "Ensure the query is well-structured, includes all relevant information needed, and is cleaned of special characters and converted to lowercase. \n\n"
        "Past Messages: {past_messages}\n"
        "User Input: {user_query}\n"
        "Construct the query in the following format:\n\n"
        "Constructed Query:\n"
    )
)

# Define a prompt for response generation
response_generation_prompt = PromptTemplate(
    input_variables=["context", "user_query", "refined query", "consulted_files", "past_messages"],
    template=(
        """You are an expert in carbon credits working for the company Tín Chỉ Carbon Việt Nam (TCCV), you are fluent in both English and Vietnamese. Based on the following context and past messages between the user and you, respond to the user's query. Try to give a long response that is as detailed and informative as possible while still being factual and on-topic. Also, respond in the same language the user query is in, regardless of the language of the context and everything else. Reference the consulted files and the DuckDuckGo search results (only those you use), reference each of them directly in the points in your response influenced by them, don't reference any one you don't use. For both types of reference, extract the title and the link, and then use hyperlink format: [title](url), for example [Tín chỉ các bon là gì?](https://vnce.vn/tin-chi-carbon-la-gi), remember to keep only the name, don't add anything like "file name", and make the hyperlink italic. If multiple consecutive points uses the same source, reference it only once at the end of those points.:\n\n"""
        "Context: {context}\n\n"
        "Past Messages: {past_messages}\n\n"
        "User Query: {user_query}\n\n"
        "Refined Query: {refined_query}\n\n"
        "Consulted Files: {consulted_files}\n\n"
        "Response:"
    )
)
def debug_step(name, inputs):
    """Debug function to print the state of variables."""
    print(f"{name}: {inputs}")
    return inputs

async def stream_llm_response(inputs) -> AsyncGenerator[str, None]:
    """Convert the synchronous iterator into an asynchronous generator."""
    
    sync_iterator = llm_for_response.stream(inputs["response_prompt"])
    
    for chunk in sync_iterator:
        await asyncio.sleep(0)  # Yield control to the event loop
        yield chunk.content  # Extract and yield the actual text content

def refine_query(inputs):
    """Refine the user query using llm_for_query and the query prompt."""
    match = re.search(
        r"Constructed Query:\s*(.+)",
        llm_for_query.predict(query_prompt_template.format(
            user_query=inputs["user_query"], 
            past_messages=inputs["past_messages"]
        ))
    )
    
    refined_query = match.group(1).strip() if match else inputs["user_query"]
    return {
        "refined_query": refined_query,
        "user_query": inputs["user_query"],
        "past_messages": inputs["past_messages"]
    }

def retrieve_file_ids(inputs):
    """Retrieve file IDs using the refined query."""

    file_info = query_for_file(
            user_query=inputs["refined_query"],
            top_k=NUM_FILE_TO_FETCH)
    return {
        "file_info": file_info,
        "user_query": inputs["user_query"],
        "refined_query": inputs["refined_query"],
        "past_messages": inputs["past_messages"]
    }

def retrieve_chunks(inputs):
    """For the retrieved file IDs, query for the best matching chunks."""
    file_ids = [file[0] for file in inputs["file_info"]]
    file_urls = [file[1] for file in inputs["file_info"]]
    return {
        "chunks": query_for_chunks(
            user_query=inputs["refined_query"],
            file_ids=file_ids,
            top_k=NUM_CHUNK_TO_FETCH_PER_FILE
        ),
        "user_query": inputs["user_query"],
        "refined_query": inputs["refined_query"],
        "consulted_files": inputs['file_info'],
        "past_messages": inputs["past_messages"]
    }

def aggregate_context(inputs):
    """Aggregate the retrieved chunks and perform a web search for additional context."""
    return {
        "context": "\n".join(inputs["chunks"]) + "\n\n" + 
                    "**Additional Context from DuckDuckGo Web Search:**\n" + 
                    str(duckduckgo_search(inputs["refined_query"], max_results=3)),
        "user_query": inputs["user_query"],
        "refined_query": inputs["refined_query"],
        "consulted_files": inputs["consulted_files"],
        "past_messages": inputs["past_messages"]
    }

def generate_response_prompt(inputs):
    """Generate the final response using the aggregated context."""
    return {
        "response_prompt": response_generation_prompt.format(
            context=inputs["context"],
            user_query=inputs["user_query"],
            refined_query=inputs["refined_query"],
            consulted_files=inputs["consulted_files"],
            past_messages=inputs["past_messages"]
        ),
        "consulted_files": inputs["consulted_files"],
        "refined_query": inputs["refined_query"]
    }

def full_workflow(user_query, past_messages):
    """Execute the entire workflow step by step."""
    inputs = {"user_query": user_query, "past_messages": past_messages}
    
    inputs = refine_query(inputs)
    # debug_step("After Query Formatting", inputs)
    
    inputs = retrieve_file_ids(inputs)
    # debug_step("After Finding Files", inputs)
    
    inputs = retrieve_chunks(inputs)
    # debug_step("After Finding Chunks", inputs)
    
    inputs = aggregate_context(inputs)
    # debug_step("After Aggregating", inputs)
    
    inputs = generate_response_prompt(inputs)
    
    return stream_llm_response(inputs)
print("First intialization")