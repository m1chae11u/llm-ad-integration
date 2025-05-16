from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import faiss
import json
import os

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.stored_texts = []
        self.stored_metadata = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()

    def build_index(self, texts: List[str], metadata: List[Dict] = None):
        """Build a FAISS index from a list of texts."""
        embeddings = self.create_embeddings(texts)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.stored_texts = texts
        self.stored_metadata = metadata if metadata is not None else [{} for _ in texts]

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts using a query."""
        if self.index is None:
            raise ValueError("No index available. Please build index first.")

        # Create query embedding
        query_embedding = self.create_embeddings([query])

        # Search index
        distances, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.stored_texts):  # Ensure valid index
                # Convert distance to similarity score (1 / (1 + distance))
                similarity_score = 1 / (1 + distance)
                results.append({
                    "text": self.stored_texts[idx],
                    "metadata": self.stored_metadata[idx],
                    "score": similarity_score
                })

        return results

    def save_index(self, path: str):
        """Save the FAISS index and associated data."""
        if self.index is None:
            raise ValueError("No index to save")

        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save texts and metadata
        with open(os.path.join(path, "data.json"), "w") as f:
            json.dump({
                "texts": self.stored_texts,
                "metadata": self.stored_metadata
            }, f)

    def load_index(self, path: str):
        """Load a saved FAISS index and associated data."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index path {path} does not exist")

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load texts and metadata
        with open(os.path.join(path, "data.json"), "r") as f:
            data = json.load(f)
            self.stored_texts = data["texts"]
            self.stored_metadata = data["metadata"] 