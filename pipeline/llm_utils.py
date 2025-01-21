import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from .embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_pinecone import PineconeVectorStore
import re
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API = os.getenv("PINECONE_API_KEY")
OPENAI_API = os.getenv("OPENAI_API_KEY")
# HUGGINGFACE_API = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API)
index = pinecone.Index("education-file-chunks")

# Use your custom embeddings class
embedding_model = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key='text')

# Chain to use the LLM with the prompt
llm_for_query = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API)
llm_for_response = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API)

def query_vectorstore(user_query: str, top_k: int = 3):
    # Step 1: Query Pinecone
    results = vectorstore.similarity_search(user_query, k=top_k)
    if not results:
        return "No relevant results found."

    print(f"Raw Query Results: {results}")
    context = '\n'.join([doc.page_content for doc in results])
    return context

# Define the prompt template
query_prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template=(
        "You are an advanced query assistant with expertise in carbon credits. "
        "Analyze the user's input to construct a meaningful, contextually complete query that can be vectorized for semantic similarity search. "
        "For clarity, you can break the input into meaningful components for easy similarity search. "
        "Ensure the query is well-structured, includes all relevant context, and is cleaned of special characters and converted to lowercase. \n\n"
        "Additionally, determine a suitable value for top_k (<= 7). This is the number of vectors (each corresponds to around 230 words) to retrieve from the database. "
        "If the query is very broad, suggest a higher top_k (5 - 7). If it is specific, suggest a lower top_k (2 - 4).\n\n"
        "User Input: {user_query}\n"
        "Construct a clean and complete query and suggest top_k in the following format:\n\n"
        "Query: <constructed query>\n"
        "Top_k: <number>"
    )
)

# Define a prompt for response generation
response_generation_prompt = PromptTemplate(
    input_variables=["context", "user_query"],
    template=(
        "You are an expert with strong expertise in carbon credits. Based on the following context, respond to the user's query:\n\n"
        "Context: {context}\n\n"
        "User Query: {user_query}\n\n"
        "Response:"
    )
)

def debug_step(name):
    """Debug function to print the state of variables."""
    return RunnableLambda(func=lambda inputs: {**inputs, "debug": print(f"{name}: {inputs}")})

query_chain = (
    RunnableLambda(
        func=lambda inputs: {
            "formatted_prompt": query_prompt_template.format(user_query=inputs["user_query"]),
            "user_query": inputs['user_query']
        }
    )
    | debug_step("After Query Formatting")
    | RunnableLambda(
        func=lambda inputs: {
            "llm_response": llm_for_query.predict(inputs["formatted_prompt"]),
            "user_query": inputs['user_query']
        }
    )
    | debug_step("After LLM Query")
    | RunnableLambda(
        func=lambda inputs: {
            # Parse the query and top_k from the LLM response
            **inputs,
            "llm_query": re.search(r"Query:\s*(.+)", inputs["llm_response"]).group(1).strip(),
            "top_k": min(6, int(re.search(r"Top_k:\s*(\d+)", inputs["llm_response"]).group(1).strip())),
        }
    )
    | debug_step("After Parsing Query and Top_k")
)

vector_search_chain = (
    RunnableLambda(
        func=lambda inputs: {
            "context": query_vectorstore(inputs['llm_query'], top_k=inputs['top_k']),
            "user_query": inputs['user_query']
        }
    )
    | debug_step("After Vector Search")
)

response_chain = (
    RunnableLambda(
        func=lambda inputs: {
            "response_prompt": response_generation_prompt.format(
                context=inputs["context"],
                user_query=inputs["user_query"]
            )
        }
    )
    | debug_step("After Response Prompt Formatting")
    | RunnableLambda(
        func=lambda inputs: {
            "response": llm_for_response.predict(inputs["response_prompt"]),
            **inputs
        }
    )
    | debug_step("Final Response")
)

full_chain = query_chain | vector_search_chain | response_chain
print("First intialization")

def generate_response(user_query: str):
    # Run the full chain
    result = full_chain.invoke(input={"user_query": user_query})
    return result["response"]