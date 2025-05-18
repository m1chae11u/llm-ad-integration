import numpy as np
from typing import List, Dict
import faiss
import json
import os
import sys
from pathlib import Path

# Add src to path to import judge.utils
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from judge.utils import batch_get_embeddings

class EmbeddingManager:
    def __init__(self):
        self.index = None
        self.stored_texts = []
        self.stored_metadata = []
        self.dimension = 1536  # text-embedding-ada-002 output size

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts using OpenAI text-embedding-ada-002."""
        embeddings = batch_get_embeddings(texts)
        return np.stack(embeddings)

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
        query_embedding = self.create_embeddings([query])
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.stored_texts):
                similarity_score = 1 / (1 + distance)
                results.append({
                    "text": self.stored_texts[idx],
                    "metadata": self.stored_metadata[idx],
                    "score": similarity_score
                })
        return results

    def save_index(self, path: str):
        if self.index is None:
            raise ValueError("No index to save")
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "data.json"), "w") as f:
            json.dump({
                "texts": self.stored_texts,
                "metadata": self.stored_metadata
            }, f)

    def load_index(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index path {path} does not exist")
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "data.json"), "r") as f:
            data = json.load(f)
            self.stored_texts = data["texts"]
            self.stored_metadata = data["metadata"] 