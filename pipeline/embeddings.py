from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from langchain.embeddings.base import Embeddings
from .utils import clean_vietnamese_text
import torch
import re
import numpy as np
from pinecone import Pinecone

# UTILS
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def normalize_embeddings(self, embeddings):
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings (numpy.ndarray): Array of embeddings (n_samples, n_features).

        Returns:
            numpy.ndarray: Normalized embeddings.
        """
        norms = np.linalg.norm(embeddings.cpu(), axis=1, keepdims=True)
        return embeddings.cpu() / norms

    def embed_query(self, text: str, norm=True) -> list:
        """Generate embeddings for a single query."""
        text = clean_vietnamese_text(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        if norm:
            embeddings = self.normalize_embeddings(embeddings)

        return embeddings[0].tolist()
    
    def embed_documents(self, texts: list, norm=True) -> list:
        """Generate embeddings for multiple documents."""
        total_embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)
            if norm:
                embeddings = self.normalize_embeddings(embeddings)
            total_embeddings.append(embeddings[0].tolist())
        return embeddings

if __name__ == "__main__":
    # TEST RUN
    pc = Pinecone(api_key="pcsk_4zpvt7_CQdf4Np2EZM2mk4cBh9CAGHrxTb9vS8Hw8hHyPcsf3Kd7mVPFbDwiPKJ1exjx27")
    index = pc.Index("education-file-chunks")

    # Check index statistics
    print(index.describe_index_stats())

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings()

    def query_pinecone(user_query: str, top_k: int = 3):
        """Query Pinecone and return the top_k results."""
        # Generate the query embedding
        query_embedding = embedding_model.embed_query(user_query)
        print(np.array(query_embedding).shape)

        # Query Pinecone
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        # Format and return results
        return [
            {"id": match["id"], "score": match["score"], "metadata": match["metadata"]}
            for match in results.get("matches", [])
        ]

    # Example usage
    results = query_pinecone("đặc điểm các loại tín chỉ carbon và tiềm năng lợi nhuận của chúng")
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}, Metadata: {result['metadata']}")
