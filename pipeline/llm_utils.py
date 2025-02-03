import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from .embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_pinecone import PineconeVectorStore
import re
import os
from dotenv import load_dotenv
from .utils import convert_bold_to_html, markdown_to_custom_html

load_dotenv()
PINECONE_API = os.getenv("PINECONE_API_KEY")
OPENAI_API = os.getenv("OPENAI_API_KEY")
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
    text_key='id'
)

# Chain to use the LLM with the prompt
llm_for_query = ChatOpenAI(model_name="gpt-4o-mini", 
                           openai_api_key=OPENAI_API)
llm_for_response = ChatOpenAI(model_name="gpt-4o-mini", 
                              openai_api_key=OPENAI_API)

def query_vectorstore(vectorstore: PineconeVectorStore, user_query: str, top_k: int = 3, **kwargs):
    # Step 1: Query Pinecone
    results = vectorstore.similarity_search(user_query, k=top_k, **kwargs)
    if not results:
        return "No relevant results found."

    return results

def query_for_file(user_query: str, top_k: int = 3):
    """
    Query the vector store for file info (summaries) and return the top_k file_ids.
    """
    query_res = query_vectorstore(file_info_vectorstore, user_query, top_k)

    file_ids = [doc.page_content for doc in query_res]
    return file_ids

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

# Define the prompt template
query_prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template=(
        "You are an advanced query assistant with expertise in carbon credits. "
        "Analyze the user's input to construct a meaningful, contextually complete query that can be vectorized for semantic similarity search. "
        "For clarity, you can break the input into clear and meaningful components for easy similarity search. "
        "Ensure the query is well-structured, includes all relevant information needed, and is cleaned of special characters and converted to lowercase. \n\n"
        "User Input: {user_query}\n"
        "Construct the query in the following format:\n\n"
        "Constructed Query:\n"
    )
)

# Define a prompt for response generation
response_generation_prompt = PromptTemplate(
    input_variables=["context", "user_query"],
    template=(
        "You are an expert with strong expertise in carbon credits. Based on the following context, respond to the user's query. Try to give a very long response that is as detailed as possible, with lots of information. And remember to be factual:\n\n"
        "Context: {context}\n\n"
        "User Query: {user_query}\n\n"
        "Response:"
    )
)

def debug_step(name):
    """Debug function to print the state of variables."""
    return RunnableLambda(func=lambda inputs: {**inputs, "debug": print(f"{name}: {inputs}")})

# New chain step: Refine the user query using llm_for_query and the query prompt.
refine_query_chain = RunnableLambda(
    func=lambda inputs: {
        "refined_query": re.search(
            r"Constructed Query:\s*(.+)",
            llm_for_query.predict(query_prompt_template.format(user_query=inputs["user_query"]))
        ).group(1).strip(),
        "user_query": inputs["user_query"]
    }
)

# Chain step 1: Retrieve file IDs (5 most relevant files) using the refined query.
file_query_chain = RunnableLambda(
    func=lambda inputs: {
        "file_ids": query_for_file(
            user_query=inputs["refined_query"],
            top_k=5  # Query 5 most relevant files
        ),
        "user_query": inputs["user_query"],
        "refined_query": inputs["refined_query"]
    }
)

# Chain step 2: For the retrieved file IDs, query for chunks (3 best matching chunks per file) using the refined query.
chunks_query_chain = RunnableLambda(
    func=lambda inputs: {
        "chunks": query_for_chunks(
            user_query=inputs["refined_query"],
            file_ids=inputs["file_ids"],
            top_k=3  # Query 3 best-matching chunks per file
        ),
        "user_query": inputs["user_query"],
        "consulted_files": [
            file_id.removeprefix("/content/drive/MyDrive/") 
            for file_id in inputs["file_ids"]
        ]
    }
)

# Chain step 3: Aggregate the chunks to form the context
aggregation_chain = RunnableLambda(
    func=lambda inputs: {
        "context": "\n".join(inputs["chunks"]),
        "user_query": inputs["user_query"],
        "consulted_files": inputs["consulted_files"]
    }
)

# Chain step 4: Generate the final response using the aggregated context and the original user query
response_chain = RunnableLambda(
    func=lambda inputs: {
        "response_prompt": response_generation_prompt.format(
            context=inputs["context"],
            user_query=inputs["user_query"]
        ),
        "consulted_files": inputs["consulted_files"]
    }
) | RunnableLambda(
    func=lambda inputs: (
        lambda response: {
            "response": response.content,
            "prompt_tokens": response.response_metadata["token_usage"]["prompt_tokens"],
            "completion_tokens": response.response_metadata["token_usage"]["completion_tokens"],
            "total_tokens": response.response_metadata["token_usage"]["total_tokens"],
            "consulted_files": inputs["consulted_files"]
        }
    )(llm_for_response.invoke(inputs["response_prompt"]))
)

# Combine all the chains to form the full workflow.
full_chain = (
    refine_query_chain
    | debug_step("After Query Formatting")
    | file_query_chain
    | debug_step("After Finding Files")
    | chunks_query_chain
    # | debug_step("After Finding Chunks")
    | aggregation_chain
    # | debug_step("After Aggregating")
    | response_chain
)

print("First intialization")

def generate_response(user_query: str):
    # Run the full chain
    result = full_chain.invoke(input={"user_query": user_query})
    response = result["response"]
    consulted_files = result["consulted_files"]
    response_ending = "\n\n Nguồn thông tin:" + "\n- " + "\n- ".join(consulted_files)

    # print token count
    print(f"Prompt Tokens: {result['prompt_tokens']}")
    print(f"Completion Tokens: {result['completion_tokens']}")
    print(f"Total Tokens: {result['total_tokens']}")
    
    final_response = response + response_ending
    return final_response