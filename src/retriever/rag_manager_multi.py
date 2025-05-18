from .embeddings import EmbeddingManager
import json
import os
import numpy as np
from typing import List, Dict, Optional

class RAGManagerMulti:
    def __init__(self, index_path: Optional[str] = None):
        # Use the same index as the original RAGManager by default
        self.embedding_manager = EmbeddingManager()
        self.index_path = index_path or "data/rag_index"
        self._load_or_create_index()
        self.min_relevance_score = 0.6  # Minimum relevance score threshold

    def _load_or_create_index(self):
        try:
            self.embedding_manager.load_index(self.index_path)
            print(f"✅ Loaded existing (shared) RAG index from {self.index_path}")
        except (ValueError, FileNotFoundError):
            print("⚠️ No existing index found. Will create new index when ads are added.")

    def add_ads_to_index(self, ads: List[Dict]):
        texts = []
        metadata = []
        for ad in ads:
            text = self._create_rich_ad_text(ad)
            texts.append(text)
            metadata.append({
                "ad_id": ad["ad_id"],
                "ad_data": ad,
                "success_rate": ad.get("success_rate", 0.5)
            })
        self.embedding_manager.build_index(texts, metadata)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.embedding_manager.save_index(self.index_path)
        print(f"✅ Added {len(ads)} ads to shared RAG index")

    def get_relevant_ads(self, query: str, k: int = 100) -> List[Dict]:
        results = self.embedding_manager.search(query, k=k)
        enhanced_results = []
        for result in results:
            ad_text = result["text"]
            ad_data = result["metadata"]["ad_data"]
            contextual_score = self._calculate_contextual_score(query, ad_text, ad_data)
            final_score = 0.7 * result["score"] + 0.3 * contextual_score
            enhanced_results.append({
                "text": ad_text,
                "ad_data": ad_data,
                "score": result["score"],
                "contextual_score": contextual_score,
                "final_score": final_score
            })
        enhanced_results.sort(key=lambda x: x["final_score"], reverse=True)
        return enhanced_results

    def _calculate_contextual_score(self, query: str, ad_text: str, ad_data: Dict) -> float:
        score = ad_data.get("success_rate", 0.5)
        if "successful_queries" in ad_data:
            for past_query in ad_data["successful_queries"]:
                if any(word in past_query.lower() for word in query.lower().split()):
                    score += 0.1
        keywords = ad_data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in query.lower():
                score += 0.1
        return min(1.0, max(0.0, score))

    def _create_rich_ad_text(self, ad: Dict) -> str:
        sections = [
            f"Product: {ad.get('ad_product', '')}",
            f"Brand: {ad.get('brand', '')}",
            f"URL: {ad.get('url', '')}",
            f"Description: {ad.get('description', '')}",
            f"Keywords: {', '.join(ad.get('keywords', []))}",
            f"Target Audience: {ad.get('target_audience', '')}",
            f"Use Cases: {', '.join(ad.get('use_cases', []))}",
            f"Success Rate: {ad.get('success_rate', 0)}",
            f"Previous Queries: {', '.join(ad.get('successful_queries', []))}"
        ]
        return "\n".join(section for section in sections if not section.endswith(": "))

    def combine_ads_for_prompt(self, ads: List[Dict], tokenizer, max_total_tokens: int = 1024) -> str:
        """Combine as many ad texts as possible for prompt insertion, up to a token limit."""
        combined = []
        total_tokens = 0
        for idx, ad in enumerate(ads, 1):
            ad_text = self._create_rich_ad_text(ad["ad_data"])
            ad_block = f"--- Ad #{idx} ---\n{ad_text}"
            ad_tokens = len(tokenizer.encode(ad_block, truncation=False))
            if total_tokens + ad_tokens > max_total_tokens:
                break
            combined.append(ad_block)
            total_tokens += ad_tokens
        return "\n\n".join(combined)

    def get_multi_ad_prompt_block(self, query: str, tokenizer, max_total_tokens: int = 1024) -> str:
        """Retrieve and format as many top ads as possible for prompt insertion, up to a token limit."""
        relevant_ads = self.get_relevant_ads(query, k=100)
        return self.combine_ads_for_prompt(relevant_ads, tokenizer, max_total_tokens=max_total_tokens) 