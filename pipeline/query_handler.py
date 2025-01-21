# from .vector_db import query_file_info, query_chunk_info
from .llm_utils import generate_response

async def handle_query(user_message):
    # Step 1: Query the first vector database
    # file_info_results = query_file_info(user_message)

    # # Step 2: Query the second vector database with results from step 1
    # if file_info_results:
    #     chunk_info_query = " ".join(file_info_results)  # Combine results meaningfully
    #     chunk_info_results = query_chunk_info(chunk_info_query)

    #     # Step 3: Use LangChain to generate a response
    #     if chunk_info_results:
    #         return generate_response(chunk_info_results, user_message)
    #     return "No relevant information found in chunk_info."
    # return "No relevant information found in file_info."

    response = generate_response(user_message)
    return response
